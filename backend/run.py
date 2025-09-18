
from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # The host must be set to '0.0.0.0' to be accessible from the network
    # The port and debug settings are now handled inside the create_app factory
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
