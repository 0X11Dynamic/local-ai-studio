//! Windows Job Object wrapper. Every backend process is placed in a job created
//! with KILL_ON_JOB_CLOSE, so if the app exits or crashes, Windows closes the
//! job handle and terminates the backend and anything it started. No orphans.

/// `CREATE_NO_WINDOW` process creation flag (no console window flash).
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation, SetInformationJobObject,
        TerminateJobObject,
    };
    use windows::core::PCWSTR;

    pub struct JobObject {
        handle: HANDLE,
    }

    // The handle is an opaque kernel object reference; Win32 job APIs are thread-safe.
    unsafe impl Send for JobObject {}
    unsafe impl Sync for JobObject {}

    impl JobObject {
        pub fn new() -> std::io::Result<Self> {
            unsafe {
                let handle = CreateJobObjectW(None, PCWSTR::null()).map_err(to_io)?;
                let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                if let Err(e) = SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) {
                    let _ = CloseHandle(handle);
                    return Err(to_io(e));
                }
                Ok(Self { handle })
            }
        }

        /// Assign a running process (by its raw process handle) to this job.
        pub fn assign_raw(&self, process_handle: *mut c_void) -> std::io::Result<()> {
            unsafe { AssignProcessToJobObject(self.handle, HANDLE(process_handle)).map_err(to_io) }
        }

        /// Kill every process in the job immediately.
        pub fn terminate(&self) -> std::io::Result<()> {
            unsafe { TerminateJobObject(self.handle, 1).map_err(to_io) }
        }
    }

    impl Drop for JobObject {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.handle);
            }
        }
    }

    fn to_io(e: windows::core::Error) -> std::io::Error {
        std::io::Error::other(e.to_string())
    }
}

#[cfg(windows)]
pub use imp::JobObject;

/// Place a tokio child process into a new kill-on-close job.
#[cfg(windows)]
pub fn contain(child: &tokio::process::Child) -> std::io::Result<JobObject> {
    let job = JobObject::new()?;
    let h = child
        .raw_handle()
        .ok_or_else(|| std::io::Error::other("process already exited"))?;
    job.assign_raw(h)?;
    Ok(job)
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dropping_the_job_kills_the_process() {
        let mut cmd = tokio::process::Command::new("powershell.exe");
        cmd.args(["-NoProfile", "-Command", "Start-Sleep -Seconds 60"]).creation_flags(CREATE_NO_WINDOW);
        let mut child = cmd.spawn().unwrap();
        let job = contain(&child).unwrap();
        drop(job); // last handle closed -> Windows kills the job's processes
        // A 60 s sleep that ends within 10 s was killed. (Windows reports exit
        // code 0 for kill-on-close, so the code itself proves nothing.)
        tokio::time::timeout(std::time::Duration::from_secs(10), child.wait())
            .await
            .expect("process should die when the job handle closes")
            .unwrap();
    }

    #[tokio::test]
    async fn terminate_kills_child_and_grandchild() {
        // PowerShell starts a second PowerShell; both must die with the job.
        let mut cmd = tokio::process::Command::new("powershell.exe");
        cmd.args([
            "-NoProfile",
            "-Command",
            "$p = Start-Process powershell -ArgumentList '-NoProfile','-Command','Start-Sleep 60' -PassThru -WindowStyle Hidden; \
             Write-Output $p.Id; [Console]::Out.Flush(); Start-Sleep 60",
        ])
        .stdout(std::process::Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW);
        let mut child = cmd.spawn().unwrap();
        let job = contain(&child).unwrap();
        use tokio::io::AsyncBufReadExt;
        let mut lines = tokio::io::BufReader::new(child.stdout.take().unwrap()).lines();
        let grandchild: u32 = tokio::time::timeout(std::time::Duration::from_secs(20), lines.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        job.terminate().unwrap();
        child.wait().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let out = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {grandchild}"), "/NH"])
            .output()
            .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(!text.contains(&grandchild.to_string()), "grandchild {grandchild} still running: {text}");
    }
}
