import logging
import firebase_admin as fba
from firebase_admin import auth
from a2a.server.agent_execution import RequestContext

from ..utils import AuthorisationTokenMissing, UnableToAuthenticateToken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PilotXBackend:
    _instance = None
    app = fba.initialize_app()

    @staticmethod
    async def authenticate(context: RequestContext):
        try:
            if context.call_context and context.call_context.state:
                if context.call_context.state.get("headers", None) is not None:
                    headers = context.call_context.state["headers"]
                    if headers.get("authorization", None) is not None:
                        token = headers["authorization"]
                        if token:
                            token = token.replace("Bearer ", "")
                            try:
                                decoded_token = auth.verify_id_token(token)
                                return True, {
                                    "type": "auth_success",
                                    "context": "Authenticated Successfully",
                                    **decoded_token,
                                }
                            except Exception as e:
                                raise UnableToAuthenticateToken(str(e))
                        else:
                            raise AuthorisationTokenMissing()
                    else:
                        raise AuthorisationTokenMissing()
                else:
                    raise AuthorisationTokenMissing()
            else:
                raise AuthorisationTokenMissing()
        except AuthorisationTokenMissing as atme:
            return False, {
                "type": "auth_error",
                "context": atme.message,
                "statusCode": 401,
            }
        except UnableToAuthenticateToken as uate:
            return False, {
                "type": "auth_error",
                "context": uate.message,
                "statusCode": 401,
            }
        except Exception as e:
            return False, {"type": "auth_error", "context": str(e), "statusCode": 401}
