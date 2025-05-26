"""
OpenSports Security Module

Enterprise-grade security and authentication system for sports analytics platform.
Includes JWT authentication, role-based access control, API security, and audit logging.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

from .auth import (
    AuthManager,
    JWTHandler,
    PasswordManager,
    SessionManager,
    TwoFactorAuth
)
from .rbac import (
    RoleBasedAccessControl,
    Permission,
    Role,
    User,
    AccessPolicy
)
from .api_security import (
    APIKeyManager,
    RateLimiter,
    RequestValidator,
    SecurityMiddleware,
    CORSHandler
)
from .encryption import (
    DataEncryption,
    FieldEncryption,
    KeyManager,
    SecureStorage
)
from .audit import (
    AuditLogger,
    SecurityEventMonitor,
    ComplianceReporter,
    ThreatDetector
)
from .oauth import (
    OAuthProvider,
    SSOManager,
    ExternalAuthHandler
)

__all__ = [
    'AuthManager',
    'JWTHandler',
    'PasswordManager',
    'SessionManager',
    'TwoFactorAuth',
    'RoleBasedAccessControl',
    'Permission',
    'Role',
    'User',
    'AccessPolicy',
    'APIKeyManager',
    'RateLimiter',
    'RequestValidator',
    'SecurityMiddleware',
    'CORSHandler',
    'DataEncryption',
    'FieldEncryption',
    'KeyManager',
    'SecureStorage',
    'AuditLogger',
    'SecurityEventMonitor',
    'ComplianceReporter',
    'ThreatDetector',
    'OAuthProvider',
    'SSOManager',
    'ExternalAuthHandler'
] 