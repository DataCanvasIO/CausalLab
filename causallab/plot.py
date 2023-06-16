import os
from datetime import datetime

from bokeh import models as M
from bokeh.io import show, output_notebook
from bokeh.plotting import column, row
from bokeh.plotting import curdoc
from bokeh.themes import Theme
from sklearn.model_selection import train_test_split

from ylearn.bayesian import _base
from ylearn.utils import logging, is_notebook
from . import utils
from .experiment import BNExperiment
from .theme import my_theme
from .view import BNInterventionView, BNTrainingView, BNEffectView, BNPredictionView, BNPropertyView
from .view import ViewItemNames, DataExplorationView, CausationView

logger = logging.get_logger(__name__)


class BNNotebookPlotter(_base.BObject):
    def __init__(self, bn):
        assert is_notebook(), f'Plot can only be displayed on notebook.'
        assert bn is not None

        self.bn = bn

    def plot(self, *, width=1200, height=800):
        output_notebook(hide_banner=True, verbose=False)
        view = BNPropertyView(bn=self.bn)
        layout = view.get_layout()
        layout.width = width
        layout.height = height
        layout.sizing_mode = 'fixed'
        return show(layout)


class BNExperimentPlotter(_base.BObject):
    root_name = 'myroot'

    def __init__(self, *, data_file, test_file, experiment_file, work_dir):
        # assert data_file is not None or experiment_file is not None

        self.data_file = data_file
        self.test_file = test_file
        self.experiment_file = experiment_file
        self.work_dir = work_dir

    def plot(self):
        if self.experiment_file is not None and len(self.experiment_file) > 0:
            # load experiment
            exp = BNExperiment.load(self.experiment_file)
        elif self.data_file is not None and len(self.data_file) > 0:
            # create experiment with data_file and test_file
            df_train, df_test = None, None

            if self.data_file is not None:
                df_train = utils.load_data(self.data_file)
            if self.test_file is not None:
                df_test = utils.load_data(self.test_file)

            if df_train is not None and df_test is None:
                df_train, df_test = train_test_split(df_train, test_size=0.3, random_state=123)

            exp = BNExperiment(
                train_data=df_train,
                test_data=df_test,
                causation=None,
                bn=None,
            )
        else:
            exp = None

        doc = curdoc()
        # print('>' * 30)
        # for k, v in doc.session_context.request.headers.items():
        #     print(k, ': ', v)
        # print('>' * 30)
        doc.theme = Theme(json=my_theme)
        doc.title = 'Causal Lab'

        if exp is None:
            layout = self.to_startup_layout()
        else:
            layout = self.to_experiment_layout(exp)

        myroot = column(layout, name=self.root_name, sizing_mode='stretch_both')
        doc.add_root(myroot)

    def to_experiment_layout(self, experiment):
        titles = ['Data', 'Discovery', 'Training', 'Causal Effect', #'Prediction', 'Intervention',
                  ]
        views = [
            DataExplorationView,  # (train_data=train_data, test_data=test_data),
            CausationView,  # (data=train_data, causation=causation),
            BNTrainingView,  # (bn=bn, data=train_data, causation=causation),
            BNEffectView,  # (bn=bn, data=test_data),
            # BNPredictionView,  # (bn=bn, data=test_data),
            # BNInterventionView,  # (bn=bn, data=train_data),
            # # BNPropertyView,  # (bn=bn, data=test_data),
        ]

        place_holder = M.Div(text='<br/>&nbsp;&nbsp;&nbsp;&nbsp;loading ...')
        panels = [
            M.TabPanel(title=title, child=column(place_holder, sizing_mode='stretch_both'))
            for title in titles]
        tabs = M.Tabs(tabs=panels, sizing_mode='stretch_both')
        last_view = None

        def create_view(i):
            nonlocal last_view

            if last_view is not None:
                if isinstance(last_view, CausationView):
                    experiment.causation = last_view.causation
                elif isinstance(last_view, (BNTrainingView, BNInterventionView)):
                    experiment.bn = last_view.bn
                    print('switch bn to', experiment.bn)

            cls = views[i]
            if cls is DataExplorationView:
                view = DataExplorationView(train_data=experiment.train_data, test_data=experiment.test_data)
            elif cls is CausationView:
                view = CausationView(data=experiment.train_data, causation=experiment.causation)
            elif cls is BNTrainingView:
                view = BNTrainingView(bn=experiment.bn, data=experiment.train_data, causation=experiment.causation)
            elif cls is BNInterventionView:
                view = cls(bn=experiment.bn, data=experiment.train_data)
            elif cls is BNEffectView or cls is BNPredictionView:
                view = cls(bn=experiment.bn, data=experiment.test_data)
            elif cls is BNPropertyView:
                view = cls(bn=experiment.bn, data=experiment.test_data)
            else:
                raise ValueError(f'???{cls}')

            last_view = view
            return view

        def on_tab_active(attr, old_value, new_value):
            idx_active = new_value
            for i, panel in enumerate(panels):
                if i == idx_active:
                    view = create_view(i)
                    layout = view.get_layout()
                    layout = self.add_graph_tool(layout, experiment)
                else:
                    layout = place_holder
                panel.child.children = [layout]

        tabs.on_change('active', on_tab_active)

        # init settings
        on_tab_active('active', -1, 0)

        return tabs

    def add_graph_tool(self, layout, experiment):
        """
        add a graph tool to save experiment
        """
        fig = layout.select_one({'name': ViewItemNames.main_graph})
        if fig is None:
            return layout

        ds_action = M.ColumnDataSource(data=dict(
            # x=[10, 10],
            # y=[0, 20],
            action=['save', 'download'],
            value=['', ''],
        ))
        js_cb_on_save = M.CustomJS(
            args=dict(ds=ds_action),
            code="""
            ds.selected.indices = [];
            ds.selected.indices = [0]; // trigger python callback
            // console.log(ds.data);
            """.strip()
        )
        js_cb_on_action = M.CustomJS(
            args=dict(ds=ds_action),
            code="""
            const indices = ds.selected.indices;
            const data = ds.data;
            if( indices.length>0){
                const idx = indices[0];
                if( data["action"][idx] == "download" ){
                    const filename = data["value"][idx];
                    console.log("download", filename);  
                    const link = document.createElement('a');
                    link.href = "../download/" + filename;
                    link.download = filename;
                    link.target = '_blank';
                    link.style.visibility = 'hidden';
                    link.dispatchEvent(new MouseEvent('click'));
                }
            }
            """.strip()
        )

        def on_action(attr, old_value, new_value):
            print('on_ds_flag_change', attr, old_value, new_value)
            assert isinstance(new_value, (list, tuple))
            if len(new_value) == 0 or ds_action.data['action'][new_value[0]] != 'save':
                # print('skip')
                return

            tag = datetime.now().strftime('%Y%m%d%H%M')
            file_name = f'experiment_{tag}.pkl.gz'
            experiment.save(os.path.join(self.work_dir, file_name))
            values = ds_action.data['value'].copy()
            values[1] = file_name
            ds_action.data['value'] = values
            ds_action.selected.indices = [1]  # trigger js callback
            logger.info(f'download experiment as file {file_name}')

        ds_action.selected.on_change('indices', on_action)
        ds_action.selected.js_on_change('indices', js_cb_on_action)

        save_tool = M.CustomAction(
            icon='save',
            description='Save Experiment',
            callback=js_cb_on_save,
        )

        fig.add_tools(save_tool)
        return layout

    def to_startup_layout(self):
        file_train_data = M.FileInput(accept=['.csv', '.parquet'], sizing_mode='stretch_width')
        file_test_data = M.FileInput(accept=['.csv', '.parquet'], sizing_mode='stretch_width')
        file_experiment = M.FileInput(accept='.pkl.gz', sizing_mode='stretch_width')

        btn_start = M.Button(label='Start', button_type="primary", align='center', )

        div_train_data_msg = M.Div()
        div_test_data_msg = M.Div()
        div_experiment_msg = M.Div()
        div_msg = M.Div(text='', sizing_mode='stretch_width')

        widgets_open = [
            M.Div(),
            M.Div(text='<b>Experiment file:<b>'),
            file_experiment,
            div_experiment_msg,
        ]
        widgets_new = [
            M.Div(),
            M.Div(text='<b>Train data:</b>'),
            file_train_data,
            div_train_data_msg,
            M.Div(),
            M.Div(text='<b>Test data:</b>'),
            file_test_data,
            div_test_data_msg,
        ]
        panel_new = M.TabPanel(title='New', child=column(widgets_new))
        panel_open = M.TabPanel(title='Open', child=column(widgets_open))
        tab = M.Tabs(tabs=[panel_new, panel_open], width=800, sizing_mode='stretch_width')

        layout = row(
            M.Div(text='', sizing_mode='stretch_width'),
            column(
                M.Div(text='<h2>Experiment</h2>'),
                tab,
                div_msg,
                btn_start,
                width=400,
            ),
            M.Div(text='', sizing_mode='stretch_width'),
            sizing_mode='stretch_width')

        ctx = {}

        def on_change_train_data(attr, old_value, new_value):
            filename = file_train_data.filename
            value = file_train_data.value
            df = utils.load_b64data(value, filename)
            div_train_data_msg.text = f'shape:{df.shape}'
            div_msg.text = ''
            ctx['train_data'] = df

        def on_change_test_data(attr, old_value, new_value):
            filename = file_test_data.filename
            value = file_test_data.value
            df = utils.load_b64data(value, filename)
            div_test_data_msg.text = f'shape:{df.shape}'
            div_msg.text = ''
            ctx['test_data'] = df

        def on_change_experiment(attr, old_value, new_value):
            # filename = file_experiment.filename
            value = file_experiment.value
            exp = BNExperiment.decode(value)
            div_experiment_msg.text = f'loaded.'
            div_msg.text = ''
            ctx['experiment'] = exp

        def on_btn_start():
            if tab.active == 0:  # new
                if 'train_data' not in ctx.keys():
                    div_msg.text = 'Not found train_data'
                    return
                train_data = ctx['train_data']
                if 'test_data' in ctx.keys():
                    test_data = ctx['test_data']
                else:
                    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=123)
                exp = BNExperiment(train_data=train_data, test_data=test_data, causation=None, bn=None)
            else:  # open
                if 'experiment' not in ctx.keys():
                    div_msg.text = 'Not found experiment'
                    return
                exp = ctx['experiment']

            doc = curdoc()
            myroot = doc.select_one({'name': self.root_name})
            exp_layout = self.to_experiment_layout(exp)
            myroot.children = [exp_layout]

        file_train_data.on_change('filename', on_change_train_data)
        file_test_data.on_change('filename', on_change_test_data)
        file_experiment.on_change('value', on_change_experiment)
        btn_start.on_click(on_btn_start)

        return layout
