from flask import Blueprint, render_template, flash, redirect, url_for
from data_submission.forms import DataUploadForm
from app import limiter, app, roles_required
from werkzeug.utils import secure_filename
import os
import logging
from flask import request
from flask_login import current_user

data_submission_blueprint = Blueprint('data_submission', __name__, template_folder='templates')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


@data_submission_blueprint.route('/data_submission', methods=['GET', 'POST'])
@roles_required('client')
@limiter.limit("5 per minute")
def data_submission():
    form = DataUploadForm()
    if form.validate_on_submit():
        # handle form submission
        file = form.upload.data
        # check if file is selected
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            target = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # save the uploaded data set to the server. Need to delete once the data is processed
            file.save(target)

            # log the event
            logging.warning('SECURITY - Data Submitted [%s], By [%s, %s, %s]',
                            filename,
                            current_user.email,
                            current_user.role.role_name,
                            request.remote_addr)

            flash('Data submitted')
            return redirect(url_for('retrain_model.retrain_model', filename=filename))
        else:
            flash('No file selected')
            return render_template('data_submission/data_submission.html', form=form)
    return render_template('data_submission/data_submission.html', form=form)
