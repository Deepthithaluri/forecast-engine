import time

from fastapi import FastAPI, Request

from backend.logger import logger


def register_middlewares(
    app: FastAPI,
) -> None:
    """
    Register application middleware.
    """

    @app.middleware("http")
    async def log_requests(
        request: Request,
        call_next,
    ):
        start_time = time.time()

        logger.info(
            "Incoming Request | %s %s",
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        process_time = (
            time.time() - start_time
        ) * 1000

        logger.info(
            "Completed Request | %s %s | Status=%s | %.2f ms",
            request.method,
            request.url.path,
            response.status_code,
            process_time,
        )

        return response