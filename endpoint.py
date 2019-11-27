from flask_script import Manager, Server
from flask_migrate import Migrate, MigrateCommand

from src import app, db

migrate = Migrate(app, db)
manager = Manager(app)

# migrations
manager.add_command('db', MigrateCommand)
server = Server(host="0.0.0.0", port=3000, threaded=True)
manager.add_command("runserver", server)

if __name__ == "__main__":
    # for debugging
    # import sys
    # sys.argv = ["self", "runserver"]
    manager.run()
