"""
Security middleware for request/response protection
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from backend.core.config import settings

logger = structlog.get_logger()


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware providing comprehensive request/response protection.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through security middleware.
        """
        start_time = time.time()

        # Security headers
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # Add custom headers
        response.headers["X-GuidedPATH-Version"] = settings.VERSION
        response.headers["X-Response-Time"] = str(time.time() - start_time)

        # Rate limiting check
        await self._check_rate_limit(request)

        # Log security events
        await self._log_security_event(request, response, start_time)

        return response

    async def _check_rate_limit(self, request: Request) -> None:
        """
        Basic rate limiting check using Redis.
        """
        # TODO: Implement Redis-based rate limiting
        # For now, just log the request
        client_ip = self._get_client_ip(request)
        logger.debug("Rate limit check", client_ip=client_ip, path=request.url.path)

    async def _log_security_event(self, request: Request, response: Response, start_time: float) -> None:
        """
        Log security-relevant information about the request.
        """
        process_time = time.time() - start_time

        # Only log slow requests or potential security events
        if process_time > 5.0 or response.status_code >= 400:
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")

            logger.info(
                "Security event logged",
                client_ip=client_ip,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                process_time=process_time,
                user_agent=user_agent[:200],  # Truncate long user agents
            )

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request.
        """
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to client host
        if request.client:
            return request.client.host

        return "unknown"


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling authentication and authorization.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through authentication middleware.
        """
        # Extract and validate JWT token if present
        # TODO: Implement JWT token validation

        response = await call_next(request)

        return response
