from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_exception_handlers(
    app: FastAPI,
) -> None:
    """
    Register global exception handlers.
    """

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request,
        exc: ValueError,
    ):
        return JSONResponse(
            status_code=400,
            content={
                "status": 400,
                "error": "Bad Request",
                "message": str(exc),
            },
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(
        request: Request,
        exc: RuntimeError,
    ):
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "error": "Runtime Error",
                "message": str(exc),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ):
        return JSONResponse(
            status_code=500,
            content={
                "status": 500,
                "error": "Internal Server Error",
                "message": str(exc),
            },
        )