"""
Docstring
"""

import matplotlib.pyplot as plt
import numpy as np
import io

from flask import Flask, send_file, make_response, request
from flask_restful import Resource, Api
from wtforms import StringField, SubmitField, SelectField, FormField
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, Length
from flask import render_template, redirect, url_for

from myplots import legal_congeners, legal_emistypes
from myplots import plot_dep, plot_dep_point, legal_congener_names
from myconfig import SECRET_KEY
import base64


app = Flask(__name__)
#api = Api(app)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
data = {}


class DataEntryDep(FlaskForm):

    congener = SelectField(label="Deposited congener:",
                           choices=legal_congener_names)

    latlon = StringField(label='Deposition sites: (lat,lon), ...',
                         # validators=[DataRequired()])
                         validators=[])
    submit = SubmitField("Submit")

    def get_latlonlist(self,):
        latlist = []
        lonlist = []
        pairs = self.latlon.data.split('),')
        for pair in pairs:
            if pair == '':
                continue
            print(pair)
            ll = pair.lstrip('(').rstrip(')').split(',')
            latlist.append(ll[0])
            lonlist.append(ll[1])

        return latlist, lonlist


def plot2serial(bytes_obj):
    return base64.b64encode(bytes_obj.getvalue()).decode()


@app.route('/dep', methods=['GET', 'POST'])
def dep():
    form = DataEntryDep()

    # Default values:
    uniform = 1.
    landfill = 1.
    wwtp = 1.
    hazwaste = 1.
    incinerator = 1.
    population = 1.
    map_url, point_url = None, None
    if form.validate_on_submit():
        print(request.form)

        uniform = request.form['uniform']
        landfill = request.form['landfill']
        wwtp = request.form['wwtp']
        hazwaste = request.form['hazwaste']
        incinerator = request.form['incinerator']
        population = request.form['population']

        cong = form.congener.data.upper()
        emis = {'uniform': float(uniform),
                'landfill': float(landfill),
                'wwtp': float(wwtp),
                'hazwaste': float(hazwaste),
                'incinerator': float(incinerator),
                'population': float(population),
                }
        # lats, lons = form.latitude.data, form.longitude.data
        # latlist, lonlist = lats.split(','), lons.split(',')
        latlist, lonlist = form.get_latlonlist()
        map_url = plot2serial(plot_dep(cong, emis))
        point_url = plot2serial(plot_dep_point(latlist, lonlist, cong, emis))
    return render_template('dep.html', form=form,
                           map_url=map_url, point_url=point_url,
                           uniform=uniform, landfill=landfill,
                           wwtp=wwtp, hazwaste=hazwaste,
                           incinerator=incinerator, population=population)


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response


if __name__ == '__main__':
    app.run(debug=True)
