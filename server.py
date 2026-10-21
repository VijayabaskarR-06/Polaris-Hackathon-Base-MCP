"""One MCP tool, ordinary JSON HTTP responses, no frontend required."""
import contextlib
import gc
import json
import logging
import os
from pathlib import Path

import anyio
import uvicorn
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from search import Catalog, INPUT_SCHEMA, OUTPUT_SCHEMA, is_empty_query

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
log = logging.getLogger('polaris.server')

server = Server('polaris-product-search', version='1.0.0')
catalog = None
TOOL = types.Tool(
    name='search_products',
    description='Search the product catalog with filters, ranking, and optional complementary recommendations.',
    inputSchema=INPUT_SCHEMA, outputSchema=OUTPUT_SCHEMA)


@server.list_tools()
async def list_tools():
    return [TOOL]


def tool_result(output):
    return types.CallToolResult(content=[types.TextContent(type='text', text=json.dumps(output))],
                                structuredContent=output, isError=False)


# catalog.prepare runs the same Draft 2020-12 input check the SDK would, with a
# precompiled validator, so the SDK's per-call re-validation is switched off.
# Returning a CallToolResult skips its output re-validation: outputs are valid
# by construction. Invalid arguments raise ValueError, which the SDK reports as
# an isError tool result.
@server.call_tool(validate_input=False)
async def call_tool(name, arguments):
    if name != 'search_products':
        raise ValueError('Unknown tool')
    args = catalog.prepare(arguments)
    if is_empty_query(args):
        # Exact numpy page in well under a millisecond: no thread hop.
        return tool_result(catalog.exact(args))
    return tool_result(await anyio.to_thread.run_sync(catalog.semantic, args))


security = TransportSecuritySettings(
    allowed_hosts=[h.strip() for h in os.getenv('MCP_ALLOWED_HOSTS',
        'localhost:*,127.0.0.1:*,[::1]:*').split(',') if h.strip()],
    allowed_origins=[o.strip() for o in os.getenv('MCP_ALLOWED_ORIGINS', '').split(',') if o.strip()],
)
sessions = StreamableHTTPSessionManager(server, json_response=True, stateless=True, security_settings=security)


@contextlib.asynccontextmanager
async def lifespan(app):
    global catalog
    path = os.getenv('CATALOG_PATH', str(Path(__file__).parent / 'data/products.json'))
    catalog = await anyio.to_thread.run_sync(Catalog.load, path)
    await anyio.to_thread.run_sync(catalog.warm_up)
    # Everything loaded so far lives for the whole run; keep the cyclic GC from
    # rescanning it during requests (a full pass over the catalog costs latency).
    gc.collect()
    gc.freeze()
    log.info('ready: %d products', len(catalog.products))
    async with sessions.run():
        yield


METHOD_NOT_ALLOWED = json.dumps({'jsonrpc': '2.0', 'id': 'server-error',
                                 'error': {'code': -32000, 'message': 'Method Not Allowed'}}).encode()


class MCPApplication:
    async def __call__(self, scope, receive, send):
        # No standalone SSE stream is offered: the brief requires 405 for GET.
        if scope['method'] not in ('POST', 'DELETE'):
            await send({'type': 'http.response.start', 'status': 405,
                        'headers': [(b'allow', b'POST, DELETE'), (b'content-type', b'application/json'),
                                    (b'content-length', str(len(METHOD_NOT_ALLOWED)).encode())]})
            await send({'type': 'http.response.body', 'body': METHOD_NOT_ALLOWED})
            return
        await sessions.handle_request(scope, receive, send)


async def health(request):
    return JSONResponse({'status': 'ok', 'products': len(catalog.products)})


app = Starlette(lifespan=lifespan, routes=[
    Route('/mcp', endpoint=MCPApplication(), methods=None),
    Route('/healthz', endpoint=health),
])


def fastest(module, name):
    try:
        __import__(module)
        return name
    except ImportError:
        return 'auto'


if __name__ == '__main__':
    # One process only: the semantic cache must be shared by every request.
    uvicorn.run(app, host=os.getenv('HOST', '0.0.0.0'), port=int(os.getenv('PORT', '8000')),
                access_log=False, log_level='warning',
                loop=fastest('uvloop', 'uvloop'), http=fastest('httptools', 'httptools'))
