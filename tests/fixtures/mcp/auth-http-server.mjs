// Test-only Streamable HTTP MCP server that requires a bearer token.
//   node auth-http-server.mjs <port> <token>
import http from "node:http";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";

const [port, token] = [Number(process.argv[2] ?? 18901), process.argv[3] ?? "secret-token"];

function makeServer() {
  const s = new McpServer({ name: "auth-fixture", version: "1.0.0" });
  s.registerTool("whoami", { description: "Say who is calling.", inputSchema: {} }, async () => ({
    content: [{ type: "text", text: "authenticated caller" }],
  }));
  s.registerTool("add", { description: "Add two numbers.", inputSchema: { a: z.number(), b: z.number() } }, async ({ a, b }) => ({
    content: [{ type: "text", text: String(a + b) }],
  }));
  return s;
}

const httpServer = http.createServer(async (req, res) => {
  if (req.headers.authorization !== `Bearer ${token}`) {
    res.writeHead(401, { "Content-Type": "application/json" }).end(JSON.stringify({ error: "unauthorized" }));
    return;
  }
  // Stateless: a fresh server + transport per request.
  const server = makeServer();
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
  res.on("close", () => {
    transport.close();
    server.close();
  });
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", async () => {
    await server.connect(transport);
    await transport.handleRequest(req, res, body ? JSON.parse(body) : undefined);
  });
});
httpServer.listen(port, "127.0.0.1", () => process.stdout.write(`auth-fixture listening on ${port}\n`));
