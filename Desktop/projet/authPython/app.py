
from flask import Flask,render_template,url_for,redirect

from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user

from flask_sqlalchemy import SQLAlchemy
import MySQLdb
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField,EmailField
from wtforms.validators import InputRequired, Length, ValidationError,EqualTo
from flask_bcrypt import Bcrypt
app = Flask(__name__)

bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost/authpython'




app.config['SECRET_KEY'] = 'thisisasecretkey'
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
db = SQLAlchemy(app)
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True,unique=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    nom = db.Column(db.String(80), nullable=False)
    email=db.Column(db.String(80), nullable=False)
    
    
class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    nom = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "nom"})
    email = EmailField(validators=[
                           InputRequired(), Length(min=4, max=80)], render_kw={"placeholder": "email"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    confirm = PasswordField(
        "Repeat password",
        validators=[
            InputRequired(),
            EqualTo("password", message="Vos mot de passe doivent correspondre."),
        ],
        render_kw={"placeholder": "Confirm your password"}
    )

    submit = SubmitField('Register')  
class LoginForm(FlaskForm):
    username = StringField('Email',validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
   

    submit = SubmitField('Login')      
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', form=form,erreur="Votre mot de passe est incorrect")
        else:
            return render_template('login.html', form=form,erreur="Votre username est incorrect")    
    return render_template('login.html', form=form)
@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')
@app.route('/register',methods=['GET','POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password,email=form.email.data,nom=form.nom.data)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
