"""
Advanced Authentication System for OpenSports

Enterprise-grade authentication with JWT tokens, secure password management,
session handling, and two-factor authentication capabilities.

Author: Nik Jois (nikjois@llamaearch.ai)
"""

import asyncio
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import jwt
import bcrypt
import pyotp
import qrcode
from io import BytesIO
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import redis
from cryptography.fernet import Fernet

from ..core.database import DatabaseManager
from ..core.cache import CacheManager
from ..core.config import Config

logger = logging.getLogger(__name__)

@dataclass
class AuthConfig:
    """Authentication configuration settings."""
    jwt_secret_key: str = "your-super-secret-jwt-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 60
    enable_2fa: bool = True
    totp_issuer: str = "OpenSports"

class PasswordManager:
    """Secure password management with hashing and validation."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        try:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength according to policy."""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def generate_secure_password(self, length: int = 12) -> str:
        """Generate a secure random password."""
        import string
        
        characters = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(characters) for _ in range(length))
        
        # Ensure password meets requirements
        if not self.validate_password_strength(password)[0]:
            return self.generate_secure_password(length)
        
        return password

class JWTHandler:
    """JWT token management for authentication."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
    
    def create_access_token(self, user_id: str, user_data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        try:
            expire = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
            payload = {
                'user_id': user_id,
                'exp': expire,
                'iat': datetime.utcnow(),
                'type': 'access',
                **user_data
            }
            
            token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
            return token
        except Exception as e:
            logger.error(f"Access token creation failed: {e}")
            raise
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        try:
            expire = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
            payload = {
                'user_id': user_id,
                'exp': expire,
                'iat': datetime.utcnow(),
                'type': 'refresh'
            }
            
            token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
            return token
        except Exception as e:
            logger.error(f"Refresh token creation failed: {e}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.config.jwt_secret_key, algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token."""
        try:
            payload = self.verify_token(refresh_token)
            if not payload or payload.get('type') != 'refresh':
                return None
            
            # Get user data for new access token
            user_id = payload['user_id']
            # In a real implementation, fetch user data from database
            user_data = {'roles': ['user']}  # Placeholder
            
            return self.create_access_token(user_id, user_data)
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None

class TwoFactorAuth:
    """Two-factor authentication using TOTP."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
    
    def generate_secret(self) -> str:
        """Generate TOTP secret for user."""
        return pyotp.random_base32()
    
    def generate_qr_code(self, user_email: str, secret: str) -> str:
        """Generate QR code for TOTP setup."""
        try:
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=user_email,
                issuer_name=self.config.totp_issuer
            )
            
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64 string
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"QR code generation failed: {e}")
            raise
    
    def verify_totp(self, secret: str, token: str) -> bool:
        """Verify TOTP token."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)
        except Exception as e:
            logger.error(f"TOTP verification failed: {e}")
            return False
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for 2FA recovery."""
        return [secrets.token_hex(4).upper() for _ in range(count)]

class SessionManager:
    """Session management with Redis backend."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self.cache = CacheManager()
    
    async def create_session(self, user_id: str, session_data: Dict[str, Any]) -> str:
        """Create new user session."""
        try:
            session_id = secrets.token_urlsafe(32)
            session_key = f"session:{session_id}"
            
            session_info = {
                'user_id': user_id,
                'created_at': datetime.utcnow().isoformat(),
                'last_activity': datetime.utcnow().isoformat(),
                **session_data
            }
            
            # Store session with expiration
            await self.cache.set(
                session_key,
                session_info,
                expire=self.config.session_timeout_minutes * 60
            )
            
            return session_id
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        try:
            session_key = f"session:{session_id}"
            session_data = await self.cache.get(session_key)
            
            if session_data:
                # Update last activity
                session_data['last_activity'] = datetime.utcnow().isoformat()
                await self.cache.set(
                    session_key,
                    session_data,
                    expire=self.config.session_timeout_minutes * 60
                )
            
            return session_data
        except Exception as e:
            logger.error(f"Session retrieval failed: {e}")
            return None
    
    async def update_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update session data."""
        try:
            session_key = f"session:{session_id}"
            session_data = await self.cache.get(session_key)
            
            if not session_data:
                return False
            
            session_data.update(update_data)
            session_data['last_activity'] = datetime.utcnow().isoformat()
            
            await self.cache.set(
                session_key,
                session_data,
                expire=self.config.session_timeout_minutes * 60
            )
            
            return True
        except Exception as e:
            logger.error(f"Session update failed: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        try:
            session_key = f"session:{session_id}"
            return await self.cache.delete(session_key)
        except Exception as e:
            logger.error(f"Session deletion failed: {e}")
            return False
    
    async def delete_user_sessions(self, user_id: str) -> int:
        """Delete all sessions for a user."""
        try:
            # This would require scanning Redis keys in a real implementation
            # For now, return 0 as placeholder
            return 0
        except Exception as e:
            logger.error(f"User sessions deletion failed: {e}")
            return 0

class AuthManager:
    """Main authentication manager coordinating all auth components."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self.db = DatabaseManager()
        self.cache = CacheManager()
        
        # Initialize components
        self.password_manager = PasswordManager(config)
        self.jwt_handler = JWTHandler(config)
        self.two_factor = TwoFactorAuth(config)
        self.session_manager = SessionManager(config)
        
        # Login attempt tracking
        self.login_attempts = {}
    
    async def register_user(self, email: str, password: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register new user."""
        try:
            # Validate password strength
            is_valid, errors = self.password_manager.validate_password_strength(password)
            if not is_valid:
                return {'success': False, 'errors': errors}
            
            # Check if user already exists
            existing_user = await self._get_user_by_email(email)
            if existing_user:
                return {'success': False, 'errors': ['User already exists']}
            
            # Hash password
            hashed_password = self.password_manager.hash_password(password)
            
            # Generate 2FA secret if enabled
            totp_secret = self.two_factor.generate_secret() if self.config.enable_2fa else None
            
            # Create user record
            user_id = await self._create_user({
                'email': email,
                'password_hash': hashed_password,
                'totp_secret': totp_secret,
                'is_2fa_enabled': False,
                'created_at': datetime.utcnow().isoformat(),
                **user_data
            })
            
            result = {
                'success': True,
                'user_id': user_id,
                'message': 'User registered successfully'
            }
            
            if totp_secret:
                qr_code = self.two_factor.generate_qr_code(email, totp_secret)
                backup_codes = self.two_factor.generate_backup_codes()
                result.update({
                    'qr_code': qr_code,
                    'backup_codes': backup_codes,
                    'setup_2fa': True
                })
            
            return result
            
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            return {'success': False, 'errors': ['Registration failed']}
    
    async def authenticate_user(self, email: str, password: str, totp_token: Optional[str] = None) -> Dict[str, Any]:
        """Authenticate user with email/password and optional 2FA."""
        try:
            # Check login attempts
            if await self._is_account_locked(email):
                return {'success': False, 'errors': ['Account temporarily locked']}
            
            # Get user
            user = await self._get_user_by_email(email)
            if not user:
                await self._record_failed_attempt(email)
                return {'success': False, 'errors': ['Invalid credentials']}
            
            # Verify password
            if not self.password_manager.verify_password(password, user['password_hash']):
                await self._record_failed_attempt(email)
                return {'success': False, 'errors': ['Invalid credentials']}
            
            # Check 2FA if enabled
            if user.get('is_2fa_enabled') and self.config.enable_2fa:
                if not totp_token:
                    return {'success': False, 'requires_2fa': True}
                
                if not self.two_factor.verify_totp(user['totp_secret'], totp_token):
                    await self._record_failed_attempt(email)
                    return {'success': False, 'errors': ['Invalid 2FA token']}
            
            # Clear failed attempts
            await self._clear_failed_attempts(email)
            
            # Create tokens and session
            user_data = {
                'email': user['email'],
                'roles': user.get('roles', ['user']),
                'permissions': user.get('permissions', [])
            }
            
            access_token = self.jwt_handler.create_access_token(user['id'], user_data)
            refresh_token = self.jwt_handler.create_refresh_token(user['id'])
            session_id = await self.session_manager.create_session(user['id'], user_data)
            
            return {
                'success': True,
                'access_token': access_token,
                'refresh_token': refresh_token,
                'session_id': session_id,
                'user': user_data
            }
            
        except Exception as e:
            logger.error(f"User authentication failed: {e}")
            return {'success': False, 'errors': ['Authentication failed']}
    
    async def logout_user(self, session_id: str) -> bool:
        """Logout user by deleting session."""
        try:
            return await self.session_manager.delete_session(session_id)
        except Exception as e:
            logger.error(f"User logout failed: {e}")
            return False
    
    async def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify access token and return user data."""
        try:
            payload = self.jwt_handler.verify_token(token)
            if payload and payload.get('type') == 'access':
                return payload
            return None
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None
    
    async def refresh_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token."""
        try:
            return self.jwt_handler.refresh_access_token(refresh_token)
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None
    
    async def enable_2fa(self, user_id: str, totp_token: str) -> bool:
        """Enable 2FA for user after verifying setup."""
        try:
            user = await self._get_user_by_id(user_id)
            if not user or not user.get('totp_secret'):
                return False
            
            if self.two_factor.verify_totp(user['totp_secret'], totp_token):
                await self._update_user(user_id, {'is_2fa_enabled': True})
                return True
            
            return False
        except Exception as e:
            logger.error(f"2FA enable failed: {e}")
            return False
    
    async def disable_2fa(self, user_id: str, password: str) -> bool:
        """Disable 2FA for user after password verification."""
        try:
            user = await self._get_user_by_id(user_id)
            if not user:
                return False
            
            if self.password_manager.verify_password(password, user['password_hash']):
                await self._update_user(user_id, {'is_2fa_enabled': False})
                return True
            
            return False
        except Exception as e:
            logger.error(f"2FA disable failed: {e}")
            return False
    
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> Dict[str, Any]:
        """Change user password."""
        try:
            user = await self._get_user_by_id(user_id)
            if not user:
                return {'success': False, 'errors': ['User not found']}
            
            # Verify old password
            if not self.password_manager.verify_password(old_password, user['password_hash']):
                return {'success': False, 'errors': ['Invalid current password']}
            
            # Validate new password
            is_valid, errors = self.password_manager.validate_password_strength(new_password)
            if not is_valid:
                return {'success': False, 'errors': errors}
            
            # Hash new password
            new_hash = self.password_manager.hash_password(new_password)
            
            # Update password
            await self._update_user(user_id, {'password_hash': new_hash})
            
            # Invalidate all user sessions
            await self.session_manager.delete_user_sessions(user_id)
            
            return {'success': True, 'message': 'Password changed successfully'}
            
        except Exception as e:
            logger.error(f"Password change failed: {e}")
            return {'success': False, 'errors': ['Password change failed']}
    
    # Helper methods for database operations
    async def _get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email from database."""
        # Placeholder implementation
        return None
    
    async def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID from database."""
        # Placeholder implementation
        return None
    
    async def _create_user(self, user_data: Dict[str, Any]) -> str:
        """Create user in database."""
        # Placeholder implementation
        return "user_id_placeholder"
    
    async def _update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user in database."""
        # Placeholder implementation
        return True
    
    async def _is_account_locked(self, email: str) -> bool:
        """Check if account is locked due to failed attempts."""
        try:
            attempts_key = f"login_attempts:{email}"
            attempts_data = await self.cache.get(attempts_key)
            
            if not attempts_data:
                return False
            
            if attempts_data['count'] >= self.config.max_login_attempts:
                lockout_time = datetime.fromisoformat(attempts_data['locked_at'])
                if datetime.utcnow() < lockout_time + timedelta(minutes=self.config.lockout_duration_minutes):
                    return True
                else:
                    # Lockout expired, clear attempts
                    await self.cache.delete(attempts_key)
                    return False
            
            return False
        except Exception as e:
            logger.error(f"Account lock check failed: {e}")
            return False
    
    async def _record_failed_attempt(self, email: str) -> None:
        """Record failed login attempt."""
        try:
            attempts_key = f"login_attempts:{email}"
            attempts_data = await self.cache.get(attempts_key) or {'count': 0}
            
            attempts_data['count'] += 1
            attempts_data['last_attempt'] = datetime.utcnow().isoformat()
            
            if attempts_data['count'] >= self.config.max_login_attempts:
                attempts_data['locked_at'] = datetime.utcnow().isoformat()
            
            await self.cache.set(attempts_key, attempts_data, expire=3600)  # 1 hour
        except Exception as e:
            logger.error(f"Failed attempt recording failed: {e}")
    
    async def _clear_failed_attempts(self, email: str) -> None:
        """Clear failed login attempts."""
        try:
            attempts_key = f"login_attempts:{email}"
            await self.cache.delete(attempts_key)
        except Exception as e:
            logger.error(f"Failed attempts clearing failed: {e}") 