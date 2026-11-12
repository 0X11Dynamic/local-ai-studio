// Test-only stdio MCP server for the advanced suite. Uses the low-level
// Server API so it can advertise a deliberately broken input schema.
//
//   huge_output   returns `kb` kilobytes of text (default 200)
//   bad_schema    advertises an invalid JSON Schema; returns its input
//   slow          sleeps `seconds`
//   unicode_echo  returns `text` unchanged (emoji, RTL, 100 KB inputs)
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const server = new Server({ name: "advanced-fixture", version: "1.0.0" }, { capabilities: { tools: {} } });

const tools = [
  { name: "huge_output", description: "Return a lot of text.", inputSchema: { type: "object", properties: { kb: { type: "integer", minimum: 1, maximum: 2048 } } } },
  // Invalid on purpose: unknown type name and `required` that is not an array.
  { name: "bad_schema", description: "A tool whose schema is broken.", inputSchema: { type: "object", properties: { x: { type: "not-a-real-type" } }, required: 5 } },
  { name: "slow", description: "Wait for a number of seconds.", inputSchema: { type: "object", properties: { seconds: { type: "integer", minimum: 0, maximum: 600 } }, required: ["seconds"] } },
  { name: "unicode_echo", description: "Return the text unchanged.", inputSchema: { type: "object", properties: { text: { type: "string" }, api_key: { type: "string" } }, required: ["text"] } },
];

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;
  const text = (t) => ({ content: [{ type: "text", text: t }] });
  switch (name) {
    case "huge_output": {
      const kb = Number(args.kb ?? 200);
      const line = "0123456789abcdefghijklmnopqrstuvwxyz-ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789\n"; // 76 B
      let s = "";
      while (s.length < kb * 1024) s += line;
      return text(`BEGIN\n${s.slice(0, kb * 1024)}\nEND`);
    }
    case "bad_schema":
      return text(`bad_schema got ${JSON.stringify(args)}`);
    case "slow": {
      const secs = Number(args.seconds ?? 1);
      await new Promise((r) => setTimeout(r, secs * 1000));
      return text(`waited ${secs}s`);
    }
    case "unicode_echo":
      return text(String(args.text ?? ""));
    default:
      return { isError: true, content: [{ type: "text", text: `unknown tool ${name}` }] };
  }
});

process.stderr.write("advanced-fixture ready\n");
await server.connect(new StdioServerTransport());
