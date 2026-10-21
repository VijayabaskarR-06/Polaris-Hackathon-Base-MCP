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

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')
# Our own diagnostics at INFO; the SDK's per-request INFO lines would cost event-loop time.
logging.getLogger('polaris').setLevel(os.getenv('LOG_LEVEL', 'INFO'))
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


# The endpoint is public and unauthenticated by design, so DNS-rebinding protection
# only adds failure modes: with it on, any Host header not listed exactly in
# MCP_ALLOWED_HOSTS (the public IP, an EC2 DNS name) gets HTTP 421 and discovery
# fails. It stays available as an opt-in.
security = TransportSecuritySettings(
    enable_dns_rebinding_protection=os.getenv('MCP_DNS_REBINDING_PROTECTION') == '1',
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


def accept_json(scope):
    """A POST must accept application/json for the SDK to answer; add it if a client omitted it."""
    headers = [(k, v) for k, v in scope['headers'] if k != b'accept']
    accept = b', '.join(v for k, v in scope['headers'] if k == b'accept')
    if b'application/json' not in accept:
        accept = b'application/json, text/event-stream' + (b', ' + accept if accept else b'')
        scope = dict(scope, headers=headers + [(b'accept', accept)])
    return scope


class MCPApplication:
    async def __call__(self, scope, receive, send):
        # No standalone SSE stream is offered: the brief requires 405 for GET.
        if scope['method'] not in ('POST', 'DELETE'):
            await send({'type': 'http.response.start', 'status': 405,
                        'headers': [(b'allow', b'POST, DELETE'), (b'content-type', b'application/json'),
                                    (b'content-length', str(len(METHOD_NOT_ALLOWED)).encode())]})
            await send({'type': 'http.response.body', 'body': METHOD_NOT_ALLOWED})
            return
        if scope['method'] == 'POST':
            scope = accept_json(scope)
        await sessions.handle_request(scope, receive, send)


async def health(request):
    return JSONResponse({'status': 'ok', 'products': len(catalog.products), 'semantic': catalog.semantic_status})


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
    # Keep-alive well above the MCP client's 5 s pool expiry: with equal timeouts a
    # request sent on a connection the server is just closing fails the session.
    uvicorn.run(app, host=os.getenv('HOST', '0.0.0.0'), port=int(os.getenv('PORT', '8000')),
                access_log=False, log_level='warning', timeout_keep_alive=75,
                loop=fastest('uvloop', 'uvloop'), http=fastest('httptools', 'httptools'))
