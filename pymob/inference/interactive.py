# visit http://127.0.0.1:8050/ in your web browser.
raise NotImplementedError(
    "The interactive backend is currently not implemented in pymob"
)

import json
import glob
import os
import numpy as np
import xarray as xr
from dash import Dash, html, dcc, Input, Output, ALL
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale, qualitative
import plotly.graph_objects as go
from dash_bootstrap_components import (
    Row, Col, Container, themes, Card, CardBody, Tab, Tabs)
from pymob.utils import store_file as store
from pymob.sims.simulation import update_parameters

CASE_STUDIES = "case_studies"
assert os.path.exists(CASE_STUDIES), "make sure you have a 'case_studies' directory in your project directory"


app = Dash(__name__, external_stylesheets=[themes.BOOTSTRAP])

models = dict()



case_studies = [os.path.split(cs)[-1] for cs in glob.glob(f"{CASE_STUDIES}/*")]

persist_kw = dict(persistence=True, persistence_type="session")
substance_drop = ["diuron", "naproxen", "diclofenac"]

# headline and some intro text
intro = Row(Col([
    html.H1(children='Interactive Simulation'),
    html.Div(children="""
        An interactive user interface for testing timepath
        simulation before going large scale simulations"""),
    html.Br(),
    html.H2(children="select a case study and a scenario"),
    html.Br(),
]))

# case study and scenario selection
case_study_selection = Row([
    Col(html.Div([
        dcc.Dropdown(
            options=case_studies, value=case_studies[0],
            id='case-study', **persist_kw),
        dcc.Dropdown(id='scenario', value="test", **persist_kw),
    ])),
    # substance selection
    Col(html.Div([
        dcc.Dropdown(options=substance_drop, id='substance-dropdown', 
            value=substance_drop[0], **persist_kw),
    ]))
])

# storage
storage = Row(Col([
    html.Div([
        dcc.Store(id='config-file'),
        dcc.Store(id='config-flat'),
        dcc.Store(id='parameter-keys'),
        dcc.Store(id='data-observed'),
        dcc.Store(id='data-simulated'),
    ])
]))
# parameter sliders
parameters = html.Div(id='parameter-input')


concentration_plot = Tab(Card(
    CardBody(
        html.Div([
            dcc.Graph(
                id='fig-concentration', 
                mathjax=True,
                style={'height': '90vh'}
            ),
        ]),
    )
), label="concentration")

simulation = Row(Col([
    html.Br(),
    Row([
        Col([
            html.H3("Set Parameters"),
            parameters
        ], width=4),
        Col([
            html.H3("Results"),
            dcc.Dropdown(id="observation-ids", multi=True, searchable=True, **persist_kw) ,
            Tabs(concentration_plot)
        ], width=8),
    ]),
]))

app.layout = Container([
    intro, 
    case_study_selection,
    simulation,
    storage,
], fluid=True)


@app.callback(
    Output("scenario", "options"),
    Input("case-study", "value")
)
def set_scenario_options(case_study):
    search = os.path.join(CASE_STUDIES, case_study, "scenarios", "*")
    scenarios = [os.path.split(s)[-1] for s in glob.glob(search)]
    return [{'label': i, 'value': i} for i in scenarios]


# handling scenario and case study loading
@app.callback(
    Output("config-file", "data"),
    Input("case-study", "value"),
    Input("scenario", "value"),
)
def load_study(case_study, scenario):
    # since this is computationally intensive, only load models once
    config = store.prepare_scenario(case_study, scenario)
    mod = store.import_package(package_path=config["simulation"]["package"])
    models[case_study] = mod

    return json.dumps(config)


@app.callback(
    Output('parameter-input', 'children'), 
    Output('config-flat', 'data'), 
    Output('parameter-keys', 'data'), 
    Input('config-file', 'data'),
    # Input('substance-dropdown', 'value')
)
def set_parameters(cfg_serialized):
    # config = store.prepare_scenario(case_study="tktd_mix", scenario="diuron_metabo_pyabc")
    config = json.loads(cfg_serialized)

    # how to clear this list again: see 
    # https://dash.plotly.com/pattern-matching-callbacks

    parameter_keys = []
    sections = []
    for key, config_section in config.items():
        section_params = list()
        flat_config = store.unnest(config_section, [])
        for param, value in flat_config:
            par_id = f"{key}.{param}"
            input_id = {"type": "parameter-value", "index": par_id}
            if isinstance(value, bool):
                inp = dcc.Input(
                    value=int(value),
                    type="number",
                    id=input_id,
                    **persist_kw
                )
            elif isinstance(value, (float, int)):
                inp = dcc.Input(
                    value=value, 
                    type="number",
                    id=input_id, 
                    **persist_kw
                )
                # sliders have the problem that they cannot be used displayed if 
                # values are too low. This behaviour is shit.
                # Im starting to favour normal inputs, because they are also 
                # independent of scale
                # or use log inputs and let sliders be on the log scale with 
                # appropriate marks 10 ** -1, .. 10 ** 2
                # and then transform all sliders back to 10 ** x
                # transformation of float to parameters that need to be int 
                # should be done inside the sim
            elif isinstance(value, str):
                inp = dcc.Input(
                    value=value,
                    type="text",
                    id=input_id,
                    **persist_kw
                )
            elif isinstance(value, (list, tuple)):
                continue

            else:
                raise NotImplementedError(
                    f"the config type {type(value)} of {par_id} cannot be parsed.")

            parcol = Row(Col([
                html.Div(param),
                inp
            ]))
            section_params.append(parcol)
            parameter_keys.append(par_id)

        tab = Tab(
            Card(CardBody(
                section_params)
        ), label=key)

        sections.append(tab)

    return (
        Tabs(children=sections), 
        json.dumps(flat_config), 
        json.dumps(parameter_keys)
    )


# load observed dataset
@app.callback(
    Output("data-observed", "data"),
    Output("observation-ids", "options"),
    Input("case-study", "value"),
    Input("config-file", "data"), # is only there to delay the callback of data
    Input("substance-dropdown", "value"),
)
def load_data(case_study, config_file, substance):
    data = models[case_study].data

    # this method needs to be specified via configuration
    obs_data = data.substance_tk_dataset(substance)
    obs_ids = list(obs_data.id.values)
    datas = store.serialize(obs_data, convert_time=True)
    return datas, [{'label': i, 'value': i} for i in obs_ids]


# simulate dataset
@app.callback(
    Output('data-simulated', 'data'),
    Input("case-study", "value"),
    Input("config-file", "data"),
    Input("parameter-keys", "data"),
    Input({"type": "parameter-value", "index": ALL}, "value"),
    Input("observation-ids", "value"),
    Input("data-observed", "data"),
)
def simulate(case_study, cfg_serialized, param_keys, param_values, obs_ids, obs_datas):
    config = update_parameters(
        json.loads(cfg_serialized), param_values, json.loads(param_keys))
    obs_data = store.deserialize(obs_datas)

    Simulation = models[case_study].sim.simulation
    config_modifier = models[case_study].sim.data_to_config
    
    # init and run sim, extract observations
    simulation_output = []
    for i in obs_ids:
        config = config_modifier(dataset=obs_data.sel(id=i), cfg=config, save_event_file=True)
        sim = Simulation(config)
        sim.run()
        sim_out = sim.experiment.get_tensor(return_time=True, return_xarray=True)
        sim_out = sim_out.assign_coords(id=i)

        simulation_output.append(sim_out)

    sim_data = xr.concat(simulation_output, dim="id")

    return store.serialize(sim_data, convert_time=True)


# finally plot figure
@app.callback(
    Output("fig-concentration", "figure"),
    Input("data-simulated", "data"),
    Input("data-observed", "data"),
    Input("observation-ids", "value"),
)
def fig_concentration(sim_datas, obs_datas, observation_ids):
    sim_data = store.deserialize(sim_datas)
    obs_data = store.deserialize(obs_datas)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    ns2h = 1e-9 / 3600
    colors = qualitative.D3
    for i, c in zip(observation_ids, colors):
        # plot simulated data
        fig.add_trace(
            go.Scatter(
                x=sim_data.time * ns2h, 
                y=sim_data.cext_diuron.sel(id=i), 
                name=f"sim {i}",
                mode="lines", line_color=c),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=sim_data.time * ns2h, 
                y=sim_data.cint_diuron.sel(id=i), 
                name=f"sim {i}",
                showlegend=False,
                mode="lines", line_color=c),
            row=2, col=1)

        # plot observed data
        fig.add_trace(
            go.Scatter(
                x=obs_data.time * ns2h, 
                y=obs_data.cext.sel(id=i), 
                name=f"obs {i}",
                mode="markers", marker_color=c),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=obs_data.time * ns2h, 
                y=obs_data.cint.sel(id=i), 
                showlegend=False,
                name=f"obs {i}",
                mode="markers", marker_color=c),
            row=2, col=1)
        # fig.update_traces(mode="markers")

    fig.update_layout(transition_duration=500, template="simple_white")

    x_kwargs = dict(
        showgrid=False,
        mirror=True,
        # range=[float(sim_data.time.min()), float(sim_data.time.max()) * ns2h]
    )
    
    y_kwargs = dict(
        showgrid=False,
        mirror=True,
    )

    fig.update_xaxes(title="Time [h]", row=2, col=1, **x_kwargs)
    fig.update_xaxes(title=None, row=1, col=1, **x_kwargs)
    fig.update_yaxes(title=r"$C_e$", row=1, col=1, **y_kwargs)
    fig.update_yaxes(title=r"$C_i$", row=2, col=1, **y_kwargs)
    return fig


    # debugging issues https://stackoverflow.com/questions/22711087/flask-importerror-no-module-named-app



if __name__ == "__main__":
    app.run_server(debug=False)
# from __future__ import print_function
# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets


# from objects.environments import World
# from objects.organisms import Daphnia
# from sims.simulation import create_sim
# from helpers.store_file import read_config


# # read parameter file
# def run_sim(
#     distribution,
#     search_rate,
#     maintenance_rate,
#     assimilation_rate,
#     egg_mass,
#     spawning_interval,
#     vgscs,
#     damage_baseline,
#     repair_init,
#     repair_decay,
#     config
#     ):

#     config = read_config(f"config/parameters/tests/{config}.json")
#     config["agents"]["parameters"]["search_rate_max"] = search_rate
#     config["agents"]["parameters"]["assimilation_rate_max"] = assimilation_rate
#     config["agents"]["parameters"]["maintenance_rate_max"] = maintenance_rate / 1e5
#     config["agents"]["parameters"]["spawning_interval"] = spawning_interval
#     config["agents"]["parameters"]["egg_dry_mass"] = egg_mass
#     config["agents"]["parameters"]["distribution"] = distribution
#     config["agents"]["parameters"]["rate_damage_baseline"] = damage_baseline / 1e12
#     config["agents"]["parameters"]["rate_repair_decay"] = repair_decay / 1e12
#     config["agents"]["structures"]["VGSodiumChannels"]["volumetric_channel_concentration"] = vgscs * 1e29
#     config["agents"]["X0"]["Repair"]["rate"] = repair_init / 1e16
#     # create simulation
#     s = create_sim(config)
#     # run the simulation
#     s.set()
#     s.run()

#     data = s.experiment.get_tensor(return_time=True)[:, 0, :]
#     s.plot_life_history(show=True, store=False)

#     print(f"{'maximum size:': <20} {round(max(data[:, 2]), 2)} mm")
#     print(f"{'total offspring:': <20} {sum(np.nan_to_num(data[:, 3]))}")
#     print(f"{'maintenance rate:': <20} {maintenance_rate / 1e5}")
#     print(f"{'damage baseline:': <20} {damage_baseline / 1e12}")
#     print(f"{'repair init:': <20} {repair_init / 1e16}")
#     print(f"{'repair decay:': <20} {repair_decay / 1e12}")
#     print(f"{'VGSCs:': <20} {vgscs * 1e29}")
       
# # store the data
# # s.store_sim_results()
# # s.store_observations_csv()


# distribution=widgets.FloatSlider(min=0.05, max=0.95, step=0.01, value=0.5)
# search_rate=widgets.FloatSlider(min=1e-5, max=2e-3, value=7e-4, step=2e-5, readout_format=".5f")
# maintenance_rate=widgets.FloatSlider(min=1e-6, max=1e-3, step=1e-5, value=8e-3, readout_format=".4f")
# assimilation_rate=widgets.FloatSlider(min=1e-5, max=5e-2, step=5e-5, value=4e-5, readout_format=".4f")
# egg_mass=widgets.FloatSlider(min=0.001, max=.05, step=0.001, value=0.018, readout_format=".3f")
# spawning_interval=widgets.IntSlider(min=1e4, max=1e6, step=1e4, value=1.4e5)
# damage_baseline=widgets.IntSlider(min=0, max=100000, step=1, value=0)
# repair_decay=widgets.IntSlider(min=0, max=100000, step=100, value=0)
# vgscs=widgets.IntSlider(min=0, max=100, step=1, value=10)
# repair_init=widgets.FloatSlider(min=0.0, max=100, step=1, value=14)
# config=widgets.Dropdown(options=[
#     "daphnia_expo_experiment", 
#     "daphnia_control_experiment"
# ])


# interact(
#     run_sim, 
#     distribution=distribution, 
#     search_rate=search_rate,
#     assimilation_rate=assimilation_rate,
#     maintenance_rate=maintenance_rate, 
#     egg_mass=egg_mass,
#     spawning_interval=spawning_interval, 
#     damage_baseline=damage_baseline,
#     vgscs=vgscs,
#     repair_init=repair_init,
#     repair_decay=repair_decay,
#     config=config
# )
