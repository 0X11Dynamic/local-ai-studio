// Storage folders (Settings > Storage, first-run setup). The choice is kept in
// storage.json in the data folder; the backend validates every path.
import { invoke } from "@tauri-apps/api/core";

export interface FolderCheck {
  path: string;
  ok: boolean;
  error: string | null;
  exists: boolean;
  free_bytes: number | null;
  /** Under 20 GB free on that drive. */
  low_space: boolean;
}

export interface StorageInfo {
  models: FolderCheck;
  runtimes: FolderCheck;
  data_dir: string;
  install_dir: string | null;
  default_models_dir: string;
  default_runtimes_dir: string;
  /** install-default | existing-install | user | "" */
  origin: string;
  runtime_count: number;
}

export const api8 = {
  getStorage: () => invoke<StorageInfo>("get_storage"),
  checkStorageFolder: (path: string) => invoke<FolderCheck>("check_storage_folder", { path }),
  /** A null folder is left alone. moveRuntimes copies the installed runtimes to the new runtimes folder. */
  setStorageDirs: (modelsDir: string | null, runtimesDir: string | null, moveRuntimes: boolean) =>
    invoke<StorageInfo>("set_storage_dirs", { modelsDir, runtimesDir, moveRuntimes }),
  openDataFolder: () => invoke<void>("open_data_folder"),
};
