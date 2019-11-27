from flask import render_template, Response, request, redirect, url_for, Blueprint, flash
from flask_login import login_required, login_user, logout_user
from src.forms import RegisterForm, LoginForm
from src import db, bcrypt, login_manager
from src.db_model import User
from src.video_feed import VideoFeed

from utils.constant import VIDEO_SRC


main_blueprint = Blueprint('main', __name__, )
vif = VideoFeed(stream_src=VIDEO_SRC)


@main_blueprint.route('/')
@login_required
def index():
    """
    Returns 'index.html' page in route '/'
    :return:
    """
    return render_template('index.html')


@main_blueprint.route('/register', methods=['GET', 'POST'])
def register():
    """
    Returns 'register.html' page in route '/register'
    Redirects 'login.html' page after stored new username and password when request method is post.
    :return:
    """
    register_form = RegisterForm(request.form)
    if request.method == 'POST' and register_form.validate_on_submit():
        user = User(register_form.username.data, register_form.email.data,
                    bcrypt.generate_password_hash(register_form.password.data).decode('utf - 8'))
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('main.login'))
    return render_template('register.html', form1=register_form)


@main_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    """
    Returns 'login.html' page in route '/login'.
    Redirects the 'index.html' page or next url when login is successful.
    :return:
    """
    login_form = LoginForm(request.form)
    if request.method == 'POST' and login_form.validate():
        user = User.query.filter_by(email=login_form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, login_form.password.data):
            login_user(user)
            next = request.args.get('next')
            return redirect(next or url_for('main.index'))

        else:
            flash('Invalid email and/or password.', 'danger')
            return render_template('login.html', login_form=login_form)

    return render_template('login.html', login_form=login_form)


@main_blueprint.route('/logout')
@login_required
def logout():
    """
    Returns 'login.html' page after user is logged out.
    :return:
    """
    logout_user()
    flash('You were logged out.', 'success')
    return redirect(url_for('main.login'))


def gen():
    """
    Start the thread to receive the streaming data from socket server and returns the image processed.
    :return:
    """
    vif.start()
    while True:
        # frame = None
        frame = vif.proc_streaming_marked()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            vif.stop()


@main_blueprint.route('/video_feed')
@login_required
def video_feed():
    """
    Returns the image data as bytes.
    :return:
    """
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@login_manager.unauthorized_handler
def unauthorized_callback():
    """
    Redirects the login page when user is unauthorized.
    :return:
    """
    return redirect('/login?next=' + request.path)
