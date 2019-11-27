# import os
# from flask import Flask
# from flask_login import LoginManager
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
#
# app = Flask(__name__)
#
# cur_path = os.path.dirname(os.path.abspath(__file__))
# par_path = os.path.join(cur_path, os.pardir)
#
# login_manager = LoginManager()
# login_manager.init_app(app)
#
# db = SQLAlchemy(app)
# bcrypt = Bcrypt(app)
#
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + par_path + '/thrudata.sqlite3'
# app.config['SECRET_KEY'] = "1254sd15gfrf1edewg1er5fdfedfedfeefe_ed"
# app.config['DEBUG'] = False
#
#
# from src.views import main_blueprint
#
#
# app.register_blueprint(main_blueprint)
#
#
# from src.db_model import User
#
#
# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.filter(User.id == int(user_id)).first()
