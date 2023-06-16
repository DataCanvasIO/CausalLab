import time
from collections import defaultdict, OrderedDict
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import torch
from bokeh import models as M
from bokeh.plotting import figure, row, column, curdoc

from ylearn.bayesian import DataLoader, SviBayesianNetwork, _base
from ylearn.utils import logging, is_notebook, nmap, to_list, calc_score
from . import utils
from .discovery import CausationHolder

logger = logging.get_logger(__name__)


def _discovery(df, algs, callback=None):
    from ylearn.sklearn_ex import general_preprocessor
    from .discovery import discoverers

    assert (isinstance(algs, (list, tuple))
            and len(algs) > 0
            and all(map(lambda a: a in discoverers.keys(), algs)))

    gp = general_preprocessor(number_scaler=True)
    data = gp.fit_transform(df)

    matrix = None
    for alg in algs:
        print('>>>>run:', alg)
        start_at = time.time()
        m = discoverers[alg](data)
        while abs(time.time() - start_at) < 3:
            time.sleep(0.1)
        if callback is not None:
            callback(alg, m)
        if matrix is None:
            matrix = m
        else:
            matrix += m

    return matrix


class ViewItemNames:
    main_graph = 'main_graph'
    main_table = 'main_table'

    graph_node = 'graph_node'


class PlotView(_base.BObject):
    """
    PlotView base class
    """

    def __init__(self, node_states):
        assert isinstance(node_states, dict)

        self.node_states = node_states

    def get_layout(self):
        side_width = 300
        main_layout = self.get_main_layout()
        side_layout = self.get_side_layout(main_layout, width=side_width)
        main_layout.sizing_mode = 'stretch_both'
        side_layout.sizing_mode = 'stretch_height'
        return row(main_layout, side_layout, sizing_mode='stretch_both')

    def get_main_layout(self):
        raise NotImplemented()

    def get_side_layout(self, main_layout, *, width):
        raise NotImplemented()

    @property
    def is_py_callback_enabled(self):
        return not is_notebook()

    def format_value(self, node, value):
        if value is None or value is np.nan or pd.isna(value):
            return ''

        state = self.node_states[node]
        if isinstance(state, _base.CategoryNodeState):
            return str(value)
        elif isinstance(value, int):
            return str(value)
        else:
            return f'{float(value):.3f}'

    @staticmethod
    def format_dict(dic, title=None):
        if len(dic) > 0:
            items = [PlotView.format_kv(k, v) for k, v in dic.items()]
        else:
            items = ['<li>&lt; None &gt;</li>']

        html = '\n'.join(['<ul>'] + items + ['</ul>'])
        if title is not None:
            html = f'<p><b>{title}</b></p>\n' + html
        return html

    @staticmethod
    def format_kv(k, v):
        # return f'<li><b>{key}:</b> {value}'
        if isinstance(v, float):
            return f'<li><b>{k}:</b>&nbsp; {v:.6f}'
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            return f'<li><b>{k}:</b>&nbsp; {v:.6f}'
        else:
            return f'<li><b>{k}:</b>&nbsp; {v}'

    @staticmethod
    def decorate_pd_html(html):
        style = """
        <style scoped>
            .dataframe {
                border-collapse: collapse;
                border-color: #E9953C;
            } 
            .dataframe th {
                align: center;
            }
            .dataframe th,td {
                padding: 5px;
                border-color: #E9953C;
            }
        </style>
        """.strip()
        return '\n'.join(['<div>', style, html, '</div>'])

    def state_to_html(self, node, state=None):
        s = self.node_states[node] if state is None else state
        if isinstance(s, _base.CategoryNodeState):
            stub = pd.Series(dict(
                kind='discrete',
                classes=s.classes.tolist(),
            ))
        else:
            stub = pd.Series(dict(
                kind='continuous',
                mean=s.mean,
                scale=s.scale,
                min=s.min,
                max=s.max,
            ))
        return PlotView.decorate_pd_html(stub.to_frame()._repr_html_())

    def node_shape(self, n):
        return 'box' if isinstance(self.node_states[n], _base.CategoryNodeState) else 'ellipse'

    @staticmethod
    def default_column_formatter(df, col, **kwargs):
        kind = df[col].dtype.kind
        if kind == 'f':
            return M.NumberFormatter(format='0,0.000', **kwargs)
        elif kind in 'iu':
            return M.NumberFormatter(format='0,0', **kwargs)
        else:
            return M.StringFormatter(**kwargs)


class DataExplorationView(PlotView):
    def __init__(self, train_data, test_data):
        super().__init__(DataLoader.state_of(train_data))

        self.train_data = train_data
        self.test_data = test_data

    def activate(self, data):
        assert data is self.train_data or data is self.test_data

    def get_main_layout(self):
        # fig = figure(toolbar_location='above',
        #              tools="pan,tap,zoom_in,zoom_out",
        #              # outline_line_color='lightgray',
        #              sizing_mode='stretch_width',
        #              height=30
        #              )

        table_layout = self.get_table_layout(self.train_data, sizing_mode='stretch_both')
        column_layout = self.get_column_layout(self.train_data, sizing_mode='stretch_both')
        tabs = [M.TabPanel(title='Detail', child=table_layout),
                M.TabPanel(title='Column', child=column_layout),
                ]
        tabs = M.Tabs(tabs=tabs, sizing_mode='stretch_both')
        # return column(fig, tabs, sizing_mode='stretch_both')
        return tabs

    def get_side_layout(self, main_layout, *, width=1000):
        radio_dataset = M.RadioGroup(labels=['Train Data', 'Test Data'], active=0)
        file_test = M.FileInput(accept='.csv,.txt,.json', multiple=False)

        def on_change_dataset(attr, old_value, new_value):
            if new_value == 0:
                data = self.train_data
            else:
                data = self.test_data
            print('data', data.shape)
            table_layout = self.get_table_layout(data, sizing_mode='stretch_both')
            column_layout = self.get_column_layout(data, sizing_mode='stretch_both')
            tabs = [M.TabPanel(title='Detail', child=table_layout),
                    M.TabPanel(title='Column', child=column_layout),
                    ]
            main_layout.tabs = tabs

        def on_change_file(attr, old_value, new_value):
            print(attr, 'new value:', len(new_value))
            print(attr, 'file name:', file_test.filename)
            print(attr, 'mime_type:', file_test.mime_type)

        file_test.on_change('value', on_change_file)
        file_test.on_change('mime_type', on_change_file)
        file_test.on_change('filename', on_change_file)
        radio_dataset.on_change('active', on_change_dataset)
        widgets = [M.Div(text='<h3>Datasets:</h3>'),
                   radio_dataset,
                   file_test,
                   ]
        return column(widgets, width=width)

    def get_table_layout(self, df, **kwargs):
        fmt = self.default_column_formatter
        table_columns = [M.TableColumn(field=c,
                                       formatter=fmt(df, col=c), )
                         for c in df.columns.tolist()]
        ds_table = M.ColumnDataSource(df)
        table = M.DataTable(columns=table_columns, source=ds_table, editable=False, **kwargs)
        return table

    def get_column_layout(self, df, **kwargs):
        state = DataLoader.state_of(df)
        columns = df.columns.tolist()
        state_html = [self.state_to_html(None, state=state[c]) for c in columns]
        df_summary = pd.DataFrame(dict(column=columns, state=state_html))

        ds_summary = M.ColumnDataSource(df_summary)
        table_columns = [M.TableColumn(field='column', formatter=M.StringFormatter()),
                         M.TableColumn(field='state', formatter=M.HTMLTemplateFormatter())]
        table = M.DataTable(columns=table_columns, source=ds_summary, editable=False,
                            row_height=150,
                            **kwargs)
        return table


class GraphPlotView(PlotView):
    """
    PlotView with graph and optional dataframe
    """

    def __init__(self, *, data=None, node_states=None):
        assert node_states is not None or data is not None

        if node_states is not None:
            super().__init__(node_states)
        else:
            super().__init__(DataLoader.state_of(data))

        self.data = data

        if data is not None:
            self.ds_table = self._to_table_ds(data)
            self.ds_node, self.ds_edge = self._to_graph_ds()
        else:
            self.ds_table = None
            self.ds_node, self.ds_edge = self._to_graph_ds()

    @property
    def data_title(self):
        return 'Test Data'

    def get_graph(self):
        raise NotImplemented()

    def _to_table_ds(self, df):
        source = M.ColumnDataSource(df.copy())
        source.selected.indices = [0]
        return source

    def _get_node_edge_layout(self, graph=None, prog=None, node_pos=None, dot_options=None):
        if graph is None:
            graph = self.get_graph()

        if prog is None:
            if len(graph.get_edges()) == 0:
                prog = 'neato'
            else:
                prog = 'dot'

        node_layout, edge_layout = graph.pydot_layout(
            prog=prog, node_pos=node_pos, dot_options=dot_options)
        return node_layout, edge_layout

    def _to_graph_ds(self):
        graph = self.get_graph()
        if graph is None:
            return None, None

        ds_table = self.ds_table

        # nodes names and values
        nodes = graph.get_nodes()
        if ds_table is not None and self.is_data_layout_enabled and len(ds_table.selected.indices) > 0:
            row_idx = ds_table.selected.indices[0]
            values = [self.format_value(n, ds_table.data[n][row_idx]) for n in nodes]
        else:
            values = [''] * len(nodes)

        # nodes and edges layout
        node_layout, edge_layout = self._get_node_edge_layout()

        # create node datasource
        ds_node = M.ColumnDataSource(data=dict(
            x=[node_layout[n]['x'] for n in nodes],
            y=[node_layout[n]['y'] for n in nodes],
            width=[node_layout[n]['width'] for n in nodes],
            height=[node_layout[n]['height'] for n in nodes],
            shape=[self.node_shape(n) for n in nodes],
            node=nodes,
            value=values,
            line_width=[1] * len(nodes),
        ))

        # create edge datasource
        edges = {k: [] for k in [
            'start', 'end', 'xs', 'ys',
            'arrow_x_start', 'arrow_y_start', 'arrow_x_end', 'arrow_y_end',
            'dash',
        ]}
        for (s, e), v in edge_layout.items():
            xs, ys = utils.smooth_line(v['x'], v['y'])
            edges['start'].append(s)
            edges['end'].append(e)
            edges['xs'].append(xs)
            edges['ys'].append(ys)
            edges['arrow_x_start'].append(xs[-2])
            edges['arrow_y_start'].append(ys[-2])
            edges['arrow_x_end'].append(xs[-1])
            edges['arrow_y_end'].append(ys[-1])
            edges['dash'].append('solid')  # default style
        ds_edge = M.ColumnDataSource(data=edges)

        def on_data_table_row_change(attr, old_value, new_value):
            logger.debug(f'on_data_table_row_change, new_value={new_value}')
            assert isinstance(new_value, (list, tuple)) and len(new_value) > 0

            if len(new_value) > 0:
                idx = new_value[0]
                nodes_ = ds_node.data['node']
                new_values = [self.format_value(n, ds_table.data[n][idx]) for n in nodes_]
                ds_node.data['value'] = new_values

        if ds_table is not None and self.is_data_layout_enabled and self.is_py_callback_enabled:
            ds_table.selected.on_change('indices', on_data_table_row_change)

        return ds_node, ds_edge

    def get_main_layout(self):
        if self.data is not None and self.is_data_layout_enabled:
            table_height = 250
            data_layout = self.get_data_layout(height=table_height)
            graph_layout = self.get_graph_layout()
            return column(graph_layout, data_layout)
        else:
            graph_layout = self.get_graph_layout()
            return graph_layout

    def get_graph_layout(self):
        # main figure
        fig = figure(title='Graph',
                     toolbar_location='above',
                     tools=[],
                     outline_line_color='lightgray',
                     sizing_mode='stretch_both',
                     match_aspect=True,
                     name=ViewItemNames.main_graph,
                     )
        # fig.toolbar.logo = None  # disable bokeh logo
        fig.axis.visible = False

        def on_change_inner_width(attr, old_value, new_value):
            print(attr, old_value, new_value)
            try:
                start = fig.x_range.start
                fig.x_range.end = start + new_value / 1.5
            except:
                pass

        def on_change_inner_height(attr, old_value, new_value):
            print(attr, old_value, new_value)
            try:
                start = fig.y_range.start
                fig.y_range.end = start + new_value / 1.5
            except:
                pass

        fig.on_change('inner_width', on_change_inner_width)
        fig.on_change('inner_height', on_change_inner_height)

        self._draw_graph_node(fig)
        self._draw_graph_text(fig)
        self._draw_graph_line(fig)
        self._config_graph(fig)

        return fig

    def _config_graph(self, fig):
        fig.add_tools(*to_list('pan,tap,zoom_in,zoom_out'))
        return fig

    def _draw_graph_node(self, fig):
        """
        draw nodes shape
        """
        ds_node = self.ds_node

        # define data views
        js_filter_ellipse = M.CustomJSFilter(code='''
        const data = source.data['shape'];
        const indices = data.map(v => v=="ellipse");
        return indices;
        ''')
        js_filter_rect = M.CustomJSFilter(code='''
        const data = source.data['shape'];
        const indices = data.map(v => v!="ellipse");
        return indices;
        ''')
        vw_node_ellipse = M.CDSView(filter=js_filter_ellipse)
        vw_node_rect = M.CDSView(filter=js_filter_rect)

        shape_options = dict(
            width='width', height='height',
            line_width='line_width',
            fill_color="#F3C797",
            line_color='#B7472A',
            # selection_color='gray',
            # set visual properties for non-selected glyphs
            nonselection_fill_alpha=0.6,
            # nonselection_fill_color="lightgray",
            # nonselection_line_color="firebrick",
            # nonselection_line_alpha=1.0,
            name=ViewItemNames.graph_node,
        )

        fig.rect(source=ds_node, view=vw_node_rect, **shape_options)
        fig.ellipse(source=ds_node, view=vw_node_ellipse, **shape_options)

        return fig

    def _draw_graph_text(self, fig):
        """
        draw node text
        """
        ds_node = self.ds_node

        text_options = dict(
            text_align='center',
            text_baseline='middle',
        )
        fig.text(text='node', source=ds_node,
                 y_offset=-10,  # text_font_style='bold',
                 **text_options)
        fig.text(text='value', source=ds_node,
                 y_offset=15,
                 **text_options)

        return fig

    def _draw_graph_line(self, fig):
        """
        draw edges
        """
        ds_edge = self.ds_edge

        fig.multi_line(source=ds_edge, line_color='#B7472A', line_dash='dash', )
        oh = M.OpenHead(line_color='#B7472A', line_width=1, size=10)
        arr = M.Arrow(end=oh,
                      x_start='arrow_x_start', y_start='arrow_y_start',
                      x_end='arrow_x_end', y_end='arrow_y_end',
                      line_color='#B7472A', line_width=1,
                      source=ds_edge,
                      # level='underlay',
                      )
        fig.add_layout(arr)

        return fig

    def get_data_layout(self, *, height=200):
        # nodes = self.ds_node.data['node']
        df = self.data

        _fmt = self.default_column_formatter
        ds_columns = self.ds_table.data.keys()
        df_columns = df.columns.tolist()
        table_columns = []
        for col in ds_columns:
            if col in df_columns:
                table_columns.append(M.TableColumn(
                    field=col,
                    formatter=_fmt(df, col),
                ))
            # if self.pred_name(node) in ds_columns:
            #     table_columns.append(M.TableColumn(
            #         field=self.pred_name(node),
            #         title=f'PredOf_{node}',
            #         formatter=_fmt(df, node, font_style='italic', text_color='blue'),
            #         visible=False,
            #     ))

        table = M.DataTable(columns=table_columns, source=self.ds_table, editable=False,
                            sizing_mode='stretch_both', )
        title = M.Div(text=f"<b>{self.data_title}</b>")

        layout = column(title, table, height=height, sizing_mode='stretch_width', )
        return layout

    @property
    def is_data_layout_enabled(self):
        return True


class CausationView(GraphPlotView):
    """
    PlotView for causation discovery
    """

    def __init__(self, data, causation=None):
        if causation is None:
            causation = CausationHolder(DataLoader.state_of(data))
        self.causation = causation

        super().__init__(data=data, node_states=causation.node_states)

    @property
    def data_title(self):
        return 'Train Data'

    def get_graph(self):
        return self.causation.graph

    def _to_graph_ds(self):
        ds_node, ds_edge = super()._to_graph_ds()

        # add edge data of weight & expert from graph
        weight = []
        expert = []
        marker_x = []
        marker_y = []
        marker = []
        graph = self.get_graph()
        for start, end, xs, ys in zip(ds_edge.data['start'], ds_edge.data['end'],
                                      ds_edge.data['xs'], ds_edge.data['ys']):
            if graph.has_edge(start, end):
                edge_data = graph[start][end]
                weight.append(edge_data['weight'])
                expert.append(edge_data['expert'])
            else:
                weight.append(0.0)
                expert.append(0)
            marker.append(0)
            marker_x.append(xs[0])
            marker_y.append(ys[0])

        ds_edge.data['weight'] = weight
        ds_edge.data['expert'] = expert
        ds_edge.data['marker'] = marker
        ds_edge.data['marker_x'] = marker_x
        ds_edge.data['marker_y'] = marker_y

        return ds_node, ds_edge

    doc_cb_move_marker = None

    def _draw_graph_line(self, fig):
        """
        draw edges with **weight** as line_width, **expert** as line_color
        """
        ds_edge = self.ds_edge

        js_filter_expert = M.CustomJSFilter(code='''
        const data = source.data['expert'];
        const indices = Array.from(data).map(v => v>0);
        return indices;
        ''')
        js_filter_not_expert = M.CustomJSFilter(code='''
        const data = source.data['expert'];
        const indices = Array.from(data).map(v => v==0);
        return indices;
        ''')
        vw_edge_expert = M.CDSView(filter=js_filter_expert)
        vw_edge_not_expert = M.CDSView(filter=js_filter_not_expert)

        fig.multi_line(source=ds_edge, view=vw_edge_expert, line_color='blue', width='weight')
        fig.multi_line(source=ds_edge, view=vw_edge_not_expert, line_color='#B7472A', width='weight')
        oh = M.OpenHead(line_color='#B7472A', line_width=1, size=10)
        arr = M.Arrow(end=oh,
                      x_start='arrow_x_start', y_start='arrow_y_start',
                      x_end='arrow_x_end', y_end='arrow_y_end',
                      line_color='#B7472A', line_width=1,
                      source=ds_edge,
                      # level='underlay',
                      )
        fig.add_layout(arr)

        fig.circle(x='marker_x', y='marker_y', radius=2,
                   line_color='#B7472A', fill_color='#B7472A',
                   source=ds_edge)

        def update_marker_xy():
            data = ds_edge.data
            marker = []
            marker_x = []
            marker_y = []
            for i, (m, xs, ys) in enumerate(zip(data['marker'], data['xs'], data['ys'])):
                m_new = m + 1 if (m + 3) < len(xs) else 0
                marker.append(m_new)
                marker_x.append(xs[m_new])
                marker_y.append(ys[m_new])
            data['marker'] = marker
            data['marker_x'] = marker_x
            data['marker_y'] = marker_y

        def setup_cb():
            if CausationView.doc_cb_move_marker is not None:
                try:
                    curdoc().remove_periodic_callback(CausationView.doc_cb_move_marker)
                except:
                    pass
                CausationView.doc_cb_move_marker = None
            cb = curdoc().add_periodic_callback(update_marker_xy, 200)
            CausationView.doc_cb_move_marker = cb

        curdoc().add_timeout_callback(setup_cb, 2000)
        return fig

    def _config_graph(self, fig):
        super()._config_graph(fig)

        node_renders = fig.select({'name': ViewItemNames.graph_node})
        if node_renders is None:
            return fig

        # add PointDrawTool to drag-drop nodes
        draw_tool = M.PointDrawTool(renderers=node_renders, drag=True, add=False, )
        fig.add_tools(draw_tool)

        # fig.toolbar.active_tap = draw_tool

        def on_data_change(attr, old_value, new_value):
            # find_moved_nodes
            eps = 0.01
            nodes_moved = []
            for node, xnew, ynew, xold, yold in \
                    zip(old_value['node'], old_value['x'], old_value['y'], new_value['x'], new_value['y']):
                if abs(xnew - xold) > eps or abs(ynew - yold) > eps:
                    nodes_moved.append(node)

            if len(nodes_moved) == 0:
                return

            # update_graph_layout
            logger.info(f'found moved nodes: {nodes_moved} , call update_graph_layout')
            node_pos = {n: (x, y)
                        for n, x, y in zip(new_value['node'], new_value['x'], new_value['y'])}
            self.update_graph_layout(
                graph=self.get_graph(), prog='neato', node_pos=node_pos,
                dot_options=dict(splines='spline'))

        self.ds_node.on_change('data', on_data_change)

        return fig

    def get_side_layout(self, main_layout, *, width):
        from causallab.discovery import discoverers

        ce_sep = ' > '

        def ce_value(cause, effect):
            return f'{cause}{ce_sep}{effect}'

        def parse_ce_value(v):
            assert isinstance(v, str) and v.find(ce_sep) > 0
            i = v.find(ce_sep)
            cause = v[:i]
            effect = v[i + len(ce_sep):]
            return cause, effect

        causation, ds_node, ds_edge = self.causation, self.ds_node, self.ds_edge

        algs = list(discoverers.keys())
        nodes = self.ds_node.data['node']

        choice_algs = M.MultiChoice(
            options=algs,
            value=algs[:1],
            # max_items=3,
            placeholder='click to select',
            sizing_mode='stretch_width',
        )
        container_algs = column(M.Div(text=''))
        btn_discovery = M.Button(
            label='Run',
            button_type="primary",
        )
        container_progress = column(M.Div(text=''))
        spinner_threshold = M.Spinner(
            title='threshold of discovery',
            value=1, step=1,
            low=1, high=max(len(causation.matrices), 1),
        )
        select_cause = M.Select(
            title='cause',
            value=nodes[0],
            options=nodes,
            sizing_mode='stretch_width',
        )
        select_effect = M.Select(
            title='effect',
            value=nodes[1],
            options=nodes,
            sizing_mode='stretch_width',
        )
        btn_add_cause_effect = M.Button(
            label='Add',
            button_type="success",
            align='end',
            width=50, sizing_mode='fixed',
        )
        select_enabled = M.MultiSelect(
            options=[ce_value(c, e) for c, e in causation.enabled],
            sizing_mode='stretch_width',
        )
        btn_remove_enabled = M.Button(
            label='',
            icon=M.BuiltinIcon('x', size="1.0em", color="white"),
            button_type="success",
            align='center',
            width=50, sizing_mode='fixed',
        )
        select_selection = M.MultiSelect(
            title='Selected:',
            description='click on graph to select causal relations',
            sizing_mode='stretch_width'
        )
        btn_add_disabled = M.Button(
            label='Del',
            button_type="success",
            align='center',
            width=50, sizing_mode='fixed',
        )
        select_disabled = M.MultiSelect(
            options=[ce_value(c, e) for c, e in causation.disabled],
            sizing_mode='stretch_width',
        )
        btn_remove_disabled = M.Button(
            label='',
            icon=M.BuiltinIcon('x', size="1.0em", color="white"),
            button_type="success",
            align='center',
            width=50, sizing_mode='fixed',
        )
        widgets = [
            M.Div(text='<h3>Causation</h3><hr/>'),
            M.Div(text='<b>&raquo; Discovery:</b>'),
            choice_algs,
            # container_algs,
            btn_discovery,
            # container_progress,
            # M.Paragraph(text=''),
            M.Div(text='<h4>&raquo; Edit</h4>'),
            spinner_threshold,
            M.Div(text='<b>Add causal relationship:</b>'),
            row(select_cause, select_effect, btn_add_cause_effect,
                width=width - 10, sizing_mode='stretch_width'),
            M.Div(text='Added:'),
            row(select_enabled, btn_remove_enabled, sizing_mode='stretch_width'),
            M.Div(text='<b>Remove causal relationship:</b>'),
            # M.Div(text='Selected:'),
            row(select_selection, btn_add_disabled, sizing_mode='stretch_width'),
            M.Div(text='Removed:'),
            row(select_disabled, btn_remove_disabled, sizing_mode='stretch_width'),
        ]

        def set_state_btn_add_cause_effect(cause=None, effect=None):
            if cause is None:
                cause = select_cause.value
            if effect is None:
                effect = select_effect.value

            btn_add_cause_effect.disabled = \
                cause == effect \
                or ce_value(cause, effect) in select_enabled.options

        def set_state_bth_remove_enabled():
            btn_remove_enabled.disabled = len(select_enabled.value) == 0

        def set_state_bth_add_disabled():
            btn_add_disabled.disabled = \
                len(select_selection.value) == 0 \
                or any(map(lambda c: c in select_disabled.options, select_selection.value))

        def set_state_bth_remove_disabled():
            btn_remove_disabled.disabled = len(select_disabled.value) == 0

        def on_change_threshold(attr, old_value, new_value):
            causation.threshold = spinner_threshold.value
            self.update_graph_layout()

        def on_change_cause(attr, old_value, new_value):
            set_state_btn_add_cause_effect(cause=new_value)

        def on_change_effect(attr, old_value, new_value):
            set_state_btn_add_cause_effect(effect=new_value)

        def on_change_enabled(attr, old_value, new_value):
            set_state_bth_remove_enabled()

        def on_change_disabled(attr, old_value, new_value):
            set_state_bth_remove_disabled()

        def on_change_selection(attr, old_value, new_value):
            set_state_bth_add_disabled()

        def on_node_change(attr, old_value, new_value):
            if layout.disabled:
                return

            if len(new_value) > 0:
                idx = new_value[0]
                node = nodes[idx]

                graph = self.causation.graph
                options = [ce_value(n, node) for n in graph.get_parents(node)] + \
                          [ce_value(node, n) for n in graph.get_children(node)]
            else:
                options = []

            select_selection.options = options
            select_selection.value = []
            set_state_bth_add_disabled()

        def on_edge_change(attr, old_value, new_value):
            if layout.disabled:
                return

            if len(new_value) > 0:
                idx = new_value[0]
                start = ds_edge.data['start'][idx]
                end = ds_edge.data['end'][idx]
                options = [ce_value(start, end)]
            else:
                options = []

            select_selection.options = options
            select_selection.value = []
            set_state_bth_add_disabled()

        def on_add_cause_effect_click():
            causation.enable(select_cause.value, select_effect.value)
            select_enabled.options = [ce_value(c, e) for c, e in causation.enabled]
            select_disabled.options = [ce_value(c, e) for c, e in causation.disabled]

            set_state_btn_add_cause_effect()
            self.update_graph_layout()

        def on_remove_enabled_click():
            for v in select_enabled.value:
                causation.remove_enabled(*parse_ce_value(v))

            select_enabled.options = [ce_value(c, e) for c, e in causation.enabled]
            select_disabled.options = [ce_value(c, e) for c, e in causation.disabled]
            select_enabled.value = []

            set_state_bth_remove_enabled()
            set_state_btn_add_cause_effect()
            self.update_graph_layout()

        def on_add_disabled_click():
            for v in select_selection.value:
                causation.disable(*parse_ce_value(v))

            select_enabled.options = [ce_value(c, e) for c, e in causation.enabled]
            select_disabled.options = [ce_value(c, e) for c, e in causation.disabled]

            set_state_bth_add_disabled()
            self.update_graph_layout()

        def on_remove_disabled_click():
            for v in select_disabled.value:
                causation.remove_disabled(*parse_ce_value(v))

            select_enabled.options = [ce_value(c, e) for c, e in causation.enabled]
            select_disabled.options = [ce_value(c, e) for c, e in causation.disabled]
            select_disabled.value = []

            set_state_bth_add_disabled()
            set_state_bth_remove_disabled()
            self.update_graph_layout()

        def on_discovery_click():
            matrices = {}
            done = []
            node_pos = {n: (x, y)
                        for n, x, y in zip(ds_node.data['node'], ds_node.data['x'], ds_node.data['y'])}
            graph_stub = deepcopy(self.get_graph())
            graph_stub.remove_edges_from(graph_stub.get_edges())

            def on_discovered(alg, matrix):
                matrices[alg] = matrix

            def on_discover_success(bn):
                done.append(1)

            def on_discover_error(e):
                done.append(0)

            def random_edge():
                start, end = 0, 0
                while start == end or graph_stub.has_edge(nodes[start], nodes[end]):
                    start, end = np.random.randint(low=0, high=len(nodes), size=2)
                return nodes[start], nodes[end]

            def update_layout():
                found = matrices.copy()
                if len(found) > 0:
                    matrices.clear()
                    print('update layout with', found.keys())
                    for alg, matrix in found.items():
                        causation.add_matrix(alg, matrix)

                if len(done) > 0:
                    curdoc().remove_periodic_callback(cb)
                    btn_discovery.disabled = False
                    self.update_graph_layout()
                else:
                    edges = graph_stub.get_edges()
                    if len(edges) * 2 > len(nodes):
                        graph_stub.remove_edges_from(edges)
                    edge_start, edge_end = random_edge()
                    graph_stub.add_edge(edge_start, edge_end, weight=1.0, expert=0)
                    self.update_graph_layout(
                        graph=graph_stub, prog='neato', node_pos=node_pos,
                        dot_options=dict(splines='spline'))

            cb = curdoc().add_periodic_callback(update_layout, 100)
            btn_discovery.disabled = True
            causation.matrices.clear()
            spinner_threshold.value = 1
            spinner_threshold.high = len(choice_algs.value)
            utils.trun(_discovery,
                       args=[self.data, choice_algs.value],
                       kwargs=dict(
                           callback=on_discovered,
                       ),
                       on_success=on_discover_success,
                       on_error=on_discover_error,
                       )

        # initialize button state
        set_state_btn_add_cause_effect()
        set_state_bth_remove_enabled()
        set_state_bth_add_disabled()
        set_state_bth_remove_disabled()

        # bind event handlers
        spinner_threshold.on_change('value', on_change_threshold)
        select_cause.on_change('value', on_change_cause)
        select_effect.on_change('value', on_change_effect)
        select_enabled.on_change('value', on_change_enabled)
        select_selection.on_change('value', on_change_selection)
        select_disabled.on_change('value', on_change_disabled)

        ds_node.selected.on_change('indices', on_node_change)
        ds_edge.selected.on_change('indices', on_edge_change)

        btn_add_cause_effect.on_click(on_add_cause_effect_click)
        btn_remove_enabled.on_click(on_remove_enabled_click)
        btn_add_disabled.on_click(on_add_disabled_click)
        btn_remove_disabled.on_click(on_remove_disabled_click)
        btn_discovery.on_click(on_discovery_click)

        # return
        layout = column(widgets, width=width)
        return layout

    def update_graph_layout(self, graph=None, prog=None, node_pos=None, dot_options=None):
        causation, ds_table, ds_node, ds_edge = self.causation, self.ds_table, self.ds_node, self.ds_edge
        if graph is None:
            graph = causation.graph
        node_layout, edge_layout = self._get_node_edge_layout(
            graph, prog=prog, node_pos=node_pos, dot_options=dot_options)

        nodes = graph.get_nodes()
        if len(ds_table.selected.indices) > 0:
            row_idx = ds_table.selected.indices[0]
            values = [self.format_value(n, ds_table.data[n][row_idx]) for n in nodes]
        else:
            values = [''] * len(nodes)

        new_node_data = dict(
            x=[node_layout[n]['x'] for n in nodes],
            y=[node_layout[n]['y'] for n in nodes],
            width=[node_layout[n]['width'] for n in nodes],
            height=[node_layout[n]['height'] for n in nodes],
            shape=[self.node_shape(n) for n in nodes],
            node=nodes,
            value=values,
        )

        new_edge_data = {k: [] for k in [
            'start', 'end', 'xs', 'ys',
            'arrow_x_start', 'arrow_y_start', 'arrow_x_end', 'arrow_y_end',
            'weight', 'expert',
            'marker', 'marker_x', 'marker_y',
        ]}
        for (s, e), v in edge_layout.items():
            xs, ys = utils.smooth_line(v['x'], v['y'])
            edge_data = graph[s][e]
            new_edge_data['start'].append(s)
            new_edge_data['end'].append(e)
            new_edge_data['xs'].append(xs)
            new_edge_data['ys'].append(ys)
            new_edge_data['arrow_x_start'].append(xs[-2])
            new_edge_data['arrow_y_start'].append(ys[-2])
            new_edge_data['arrow_x_end'].append(xs[-1])
            new_edge_data['arrow_y_end'].append(ys[-1])
            new_edge_data['weight'].append(edge_data['weight'])
            new_edge_data['expert'].append(edge_data['expert'])
            new_edge_data['marker'].append(0)
            new_edge_data['marker_x'].append(xs[0])
            new_edge_data['marker_y'].append(ys[0])

        self.patch_ds(ds_node, new_node_data)
        self.patch_ds(ds_edge, new_edge_data)

    def patch_ds(self, ds, new_data):
        assert isinstance(ds, M.ColumnDataSource)
        assert isinstance(new_data, dict)

        n_old = len(next(iter(ds.data.values())))
        n_new = len(next(iter(new_data.values())))

        if n_old == n_new:
            for k, v in new_data.items():
                ds.data[k] = v
        else:
            ds.data = pd.DataFrame(new_data)

    @property
    def is_data_layout_enabled(self):
        # return False
        return True


class BNPlotView(GraphPlotView):
    """
    PlotView with BayesianNetwork
    """

    def __init__(self, *, bn=None, data=None):
        assert bn is not None or data is not None

        self.bn = bn

        if bn is not None:
            super().__init__(data=data, node_states=bn.state_)
        else:
            super().__init__(data=data, node_states=DataLoader.state_of(data))

    def get_graph(self):
        return self.bn.graph if self.bn is not None else None

    def _to_graph_ds(self):
        ds_node, ds_edge = super()._to_graph_ds()
        if ds_node is None or ds_edge is None:
            return ds_node, ds_edge

        bn = self.bn

        # nodes names and values
        # nodes = graph.get_nodes(True)
        nodes = ds_node.data['node']
        # values, predicted_values = self._get_node_values(nodes, ds_table=ds_table, idx=None)

        # node states
        state_html = [self.state_to_html(n) for n in nodes]
        ds_node.data['state'] = state_html

        # module
        module = [bn.model_.get_node_function_cls(n).__name__ if bn is not None else ''
                  for n in nodes]
        ds_node.data['module'] = module

        # upstream
        upstream_html, upstream_num = nmap(self.parent_to_html, nodes)
        ds_node.data['upstream'] = upstream_html
        ds_node.data['upstream_n'] = upstream_num

        # fitted params
        params = self.get_node_params()
        params_html = [self.params_to_html(params[n]) for n in nodes]
        ds_node.data['params'] = params_html

        # intervened
        interventions = self.get_bn_interventions()
        intervention_values = [self.format_value(n, interventions.get(n)) for n in nodes]
        ds_node.data['intervention'] = intervention_values
        ds_node.data['line_width'] = [self._line_width(n in interventions.keys()) for n in nodes]
        ds_edge.data['dash'] = [
            self._line_dash(e in interventions.keys()) for e in ds_edge.data['end']
        ]

        def on_node_data_change(attr, old_value, new_value):
            if logger.is_debug_enabled():
                logger.debug(f'on_node_data_change, new_value={new_value}')

            # update edge datasource 'dash' when node is intervened
            intervened_new = dict(zip(nodes, new_value['intervention']))
            ends = ds_edge.data['end']
            dash_old = ds_edge.data['dash']
            dash_new = [self._line_dash(intervened_new[e] != '') for e in ends]
            dash_patches = [(i, dn) for i, (do, dn) in enumerate(zip(dash_old, dash_new)) if do != dn]
            if len(dash_patches) > 0:
                ds_edge.patch(patches=dict(dash=dash_patches))

        if self.is_py_callback_enabled:
            ds_node.on_change('data', on_node_data_change)

        return ds_node, ds_edge

    def get_node_params(self):
        bn = self.bn
        params = defaultdict(OrderedDict)
        if bn is not None and bn._is_fitted:
            for k, v in bn.fitted_params.items():
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                ks = k.split('__')
                node, param_name = ks[0], '.'.join(ks[1:])
                params[node][param_name] = v
        return params

    def get_bn_interventions(self):
        bn = self.bn

        if bn is not None and bn._is_fitted:
            return bn.interventions
        else:
            return {}

    @staticmethod
    def params_to_html(params_dict):
        r = ['<div>']
        for k, v in params_dict.items():
            r.append(f'<b> &raquo; {k}:</b>')
            if isinstance(v, np.ndarray) and v.ndim == 2:
                r.append(BNPlotView.decorate_pd_html(pd.DataFrame(v)._repr_html_()))
            elif isinstance(v, np.ndarray) and v.ndim == 1:
                r.append(f'<p>')
                r.append('[ ' + ', '.join(map(lambda vi: f'{vi:.6f}', v.tolist())) + ' ]')
                r.append(f'</p>')
            else:  # scalar
                r.append(f'<p>')
                r.append(f'{v:.6f}')
                r.append(f'</p>')
        r.append('</div')
        return '\n'.join(r)

    def parent_to_html(self, node):
        parents = self.get_graph().get_parents(node)
        if len(parents) > 0:
            html = '<p>' + ', '.join(parents) + '</p>'
        else:
            html = '<p> &lt;None&gt; </p>'
        return html, len(parents)

    def summary_html(self):
        graph = self.get_graph()
        params = self.get_node_params()

        n_params = np.sum([len(nps) for nps in params.values()])
        n_elements = np.sum([np.prod(v.shape) for nps in params.values() for v in nps.values()])

        html = ['<b>Graph summary</b>']
        html.extend([
            '<ul>',
            f'<li><b>Node number:</b> {len(graph.get_nodes())}</li>',
            f'<li><b>Edge number:</b> {len(graph.get_edges())}</li>',
            f'<li><b>Param number:</b> {int(n_params)}</li>',
            f'<li><b>Param elements:</b> {int(n_elements)}</li>',
            '</ul>',
        ])

        return '\n'.join(html)

    @staticmethod
    def _line_dash(intervened):
        return 'dashed' if intervened else 'solid'

    @staticmethod
    def _line_width(intervened):
        return 2 if intervened else 1

    def _get_table_columns(self):
        _fmt = self.default_column_formatter
        nodes = self.ds_node.data['node']
        df = self.data
        ds_columns = self.ds_table.data.keys()
        table_columns = []
        for node in nodes:
            if node in ds_columns:
                table_columns.append(M.TableColumn(
                    field=node,
                    formatter=_fmt(df, node),
                ))
        return table_columns

    def get_table_layout(self, **kwargs):
        table_columns = self._get_table_columns()
        table = M.DataTable(columns=table_columns, source=self.ds_table, editable=False, **kwargs)
        title = M.Div(text=f"<b>{self.data_title}</b>")

        layout = column(title, table)
        return layout

    def _draw_graph_text(self, fig):
        """
        draw node text
        """
        ds_node = self.ds_node

        js_filter_intervened = M.CustomJSFilter(code='''
        const intervention = source.data['intervention'];
        const indices = intervention.map(v => v.length>0);
        return indices;
        ''')
        js_filter_not_intervened = M.CustomJSFilter(code='''
        const intervention = source.data['intervention']; 
        const indices = intervention.map( (v,i) => v.length==0);
        return indices;
        ''')
        vw_node_intervened = M.CDSView(filter=js_filter_intervened)
        vw_node_not_intervened = M.CDSView(filter=js_filter_not_intervened)

        text_options = dict(
            text_align='center',
            text_baseline='middle',
        )

        # 2 elements for vw_node_not_intervened
        fig.text(text='node', source=ds_node, view=vw_node_not_intervened,
                 y_offset=-10,  # text_font_style='bold',
                 **text_options)
        fig.text(text='value', source=ds_node, view=vw_node_not_intervened,
                 y_offset=15,
                 **text_options)

        # 3 elements for vw_node_intervened
        fig.text(text='node', source=ds_node, view=vw_node_intervened,
                 y_offset=-15,  # text_font_style='bold',
                 **text_options)
        fig.text(text='intervention', source=ds_node, view=vw_node_intervened,
                 y_offset=10,
                 **text_options)
        fig.text(text='value', source=ds_node, view=vw_node_intervened,
                 y_offset=25, text_alpha=0.5, text_font_size='12px',
                 **text_options)

        return fig


class BNTrainingView(BNPlotView):
    def __init__(self, *, bn=None, data=None, causation=None):
        self.causation = causation

        super().__init__(bn=bn, data=data)

    def get_graph(self):
        return self.causation.graph

    def get_side_layout(self, main_layout, *, width):
        graph = self.get_graph()
        if not graph.is_dag:
            msg = M.Div(text='<h3>Warning</h3><br>'
                             'The graph is not a valid DAG.')
            return column([msg], width=width)

        # slider_epochs = M.Slider(
        #     title='epochs',
        #     value=100, start=1, end=1000, step=1,
        #     sizing_mode='stretch_width',
        # )
        # slider_lr = M.Slider(
        #     title='learning_rate',
        #     value=0.01, start=0.001, end=0.3, step=0.001, format='.000',
        #     sizing_mode='stretch_width',
        # )
        num_epochs = M.NumericInput(
            title='epochs:',
            mode='int',
            value=100, low=1, high=1000,
            sizing_mode='stretch_width',
        )
        num_lr = M.NumericInput(
            title='learning_rate:',
            mode='float',
            value=0.01, low=0.001, high=0.5,
            sizing_mode='stretch_width',
        )
        choice_loss = M.Select(
            title='loss:',
            options=['ELBO', 'CausalEffect_ELBO'],
            sizing_mode='stretch_width',
        )
        btn_fit = M.Button(
            label='Fit',
            button_type="primary",
        )
        container_progress = column(M.Div(text=''), sizing_mode='stretch_width')
        widgets = [
            M.Div(text='<b>Settings:</b>'),
            # slider_epochs,
            # slider_lr,
            num_epochs,
            num_lr,
            choice_loss,
            M.Paragraph(text=''),
            btn_fit,
            container_progress
        ]

        def get_fitting_plot(epochs):
            source = M.ColumnDataSource(data=dict(i=[], loss=[]))
            fig = figure(title='loss:',
                         toolbar_location=None,
                         # tools="pan,tap",
                         outline_line_color='lightgray',
                         x_range=(0, epochs),
                         height=width,
                         sizing_mode='stretch_width',
                         )
            container_progress.children = [fig, ]

            fig.line(x='i', y='loss', source=source)
            return fig, source

        def on_btn_click():
            fig, ds = get_fitting_plot(num_epochs.value)
            losses = []
            done = []

            def on_fitting(i, loss):
                # print('>>>', i, loss)
                losses.append([i, loss])

            def update_fitting():
                new_data = losses.copy()
                losses.clear()

                if len(new_data) > 0:
                    i, loss = zip(*new_data)
                    patches = dict(
                        i=i,
                        loss=loss,
                    )
                    ds.stream(patches)
                    # print('stream', len(i))

                if done:
                    curdoc().remove_periodic_callback(cb)
                    btn_fit.disabled = False

            def on_fit_success(bn):
                done.append(1)
                self.bn = bn
                print(bn)

            def on_fit_error(e):
                done.append(0)
                print(e)

            cb = curdoc().add_periodic_callback(update_fitting, 100)
            print(cb)
            btn_fit.disabled = True

            bn = SviBayesianNetwork(graph)
            utils.trun(lambda: bn.fit(self.data,
                                      epochs=num_epochs.value,
                                      lr=num_lr.value,
                                      celoss=choice_loss.value == 'CausalEffect_ELBO',
                                      inplace=False,
                                      verbose=on_fitting,
                                      random_state=123,
                                      ),
                       on_success=on_fit_success,
                       on_error=on_fit_error,
                       )

        btn_fit.on_click(on_btn_click)

        layout = column(widgets, width=width)

        return layout


class FittedBNPlotView(BNPlotView):
    """
    PlotView with fitted BayesianNetwork
    """

    def get_layout(self):
        bn = self.bn
        if bn is not None and bn._is_fitted:
            return super().get_layout()
        else:
            msg = M.Div(text='<h3>Warning</h3><br>'
                             'The model is not available.')
            return column([msg])


class BNPropertyView(FittedBNPlotView):
    def get_side_layout(self, main_layout, *, width):
        summary = M.Div(text=self.summary_html())

        header = M.Div(text='<b>Node name:</b>')
        state = M.Div(text='...')
        module = M.Div(text='<b>Module:</b>')

        params_content = M.Div(text='...')  # style={'background': '#dddddd'})

        node_properties = column([
            header,
            state,
            module,
            M.Div(text='''<hr/><b>Parameters:</b>'''),
            params_content
        ], visible=False)

        ds_node = self.ds_node
        js_on_node_change = M.CustomJS(
            args=dict(ds_node=ds_node,
                      div_summary=summary, div_props=node_properties,
                      div_header=header, div_state=state,
                      div_module=module, div_params=params_content),
            code='''
            const node_selected = cb_obj.indices.length > 0;
            div_props.visible = node_selected;
            div_summary.visible = !node_selected;

            if(node_selected){
                const idx = cb_obj.indices[0];
                const data = ds_node.data;
                div_header.text = '<b> Node '+ data['node'][idx] +':</b>';
                div_module.text = '<b> Module:</b><p> &raquo; '+ data['module'][idx] +'</p>';
                div_state.text = data['state'][idx];
                div_params.text = data['params'][idx];
            }
            '''.strip()
        )
        ds_node.selected.js_on_change('indices', js_on_node_change)

        layout = column(summary, node_properties, width=width)
        return layout


class BNPredictionView(FittedBNPlotView):
    PPV = '_pred_of_'  # prefix of predicted values

    def pred_name(self, node):
        return f'{self.PPV}{node}'

    def _to_table_ds(self, df):
        # graph = bn.graph
        # nodes = graph.get_nodes()
        # outcome_nodes = filter(lambda n: len(graph.get_parents(n)) > 0, nodes)

        df = df.copy()
        for n in df.columns.tolist():  # outcome_nodes:
            df[self.pred_name(n)] = np.nan
        source = M.ColumnDataSource(df)
        return source

    def _to_graph_ds(self):
        ds_table = self.ds_table
        ds_node, ds_edge = super()._to_graph_ds()

        if ds_node is None or ds_edge is None:
            return ds_node, ds_edge

        # append data item: 'predictive'
        nodes = ds_node.data['node']
        ds_node.data['predictive'] = [''] * len(nodes)

        def on_data_table_change(attr, old_value, new_value):
            logger.debug(f'on_data_table_change, new_value={new_value}')

            # update node 'predictive' when predicted value is ready
            idx = ds_table.selected.indices[0] if len(ds_table.selected.indices) > 0 else 0
            _, predicted_values_new = self._get_node_values(nodes, ds_table=new_value, idx=idx)
            predicted_values_old = ds_node.data['predictive']
            value_pairs = zip(predicted_values_old, predicted_values_new)
            data_patches = [(i, vn) for i, (vo, vn) in enumerate(value_pairs) if vo != vn]
            if len(data_patches) > 0:
                ds_node.patch(patches=dict(predictive=data_patches))

        if ds_table is not None and self.is_py_callback_enabled:
            self.ds_table.on_change('data', on_data_table_change)

        return ds_node, ds_edge

    def _get_table_columns(self):
        _fmt = self.default_column_formatter

        nodes = self.ds_node.data['node']
        df = self.data
        ds_columns = self.ds_table.data.keys()
        table_columns = []
        for node in nodes:
            if node in ds_columns:
                table_columns.append(M.TableColumn(
                    field=node,
                    formatter=_fmt(df, node),
                ))
            if self.pred_name(node) in ds_columns:
                table_columns.append(M.TableColumn(
                    field=self.pred_name(node),
                    title=f'PredOf_{node}',
                    formatter=_fmt(df, node, font_style='italic', text_color='blue'),
                    visible=False,
                ))
        return table_columns

    def _get_node_values(self, nodes, ds_table=None, idx=None):
        assert ds_table is None or isinstance(ds_table, (M.DataSource, dict))

        if ds_table is not None:
            if isinstance(ds_table, M.DataSource):
                table_data = ds_table.data
            else:
                table_data = ds_table
            if idx is None:
                if len(ds_table.selected.indices) > 0:
                    idx = ds_table.selected.indices[0]
                else:
                    idx = 0  # use 1st line
            values = [self.format_value(n, table_data[n][idx]) for n in nodes]
            pred_names = map(self.pred_name, nodes)
            predicted_values = [
                self.format_value(n, table_data[pn][idx]) if pn in table_data.keys() else ''
                for n, pn in zip(nodes, pred_names)
            ]
        else:
            values = [''] * len(nodes)
            predicted_values = [''] * len(nodes)

        return values, predicted_values

    def _draw_graph_text(self, fig):
        """
        draw node text
        """
        ds_node = self.ds_node

        js_filter_intervened = M.CustomJSFilter(code='''
        const intervention = source.data['intervention'];
        const indices = intervention.map(v => v.length>0);
        return indices;
        ''')
        js_filter_predicted = M.CustomJSFilter(code='''
        const intervention = source.data['intervention'];
        const predictive = source.data['predictive'];
        const indices = predictive.map( (v,i) => v.length>0 && intervention[i].length==0);
        return indices;
        ''')
        js_filter_not_intervened = M.CustomJSFilter(code='''
        const intervention = source.data['intervention'];
        const predictive = source.data['predictive'];
        const indices = intervention.map( (v,i) => v.length==0 && predictive[i].length==0);
        return indices;
        ''')
        vw_node_intervened = M.CDSView(filter=js_filter_intervened)
        vw_node_predicted = M.CDSView(filter=js_filter_predicted)
        # vw_node_intervened_or_predicted = \
        #     M.CDSView(filter=M.UnionFilter(operands=[js_filter_intervened, js_filter_predicted]))
        vw_node_not_intervened = M.CDSView(filter=js_filter_not_intervened)

        text_options = dict(
            text_align='center',
            text_baseline='middle',
        )
        fig.text(text='node', source=ds_node, view=vw_node_not_intervened,
                 y_offset=-10,  # text_font_style='bold',
                 **text_options)
        fig.text(text='value', source=ds_node, view=vw_node_not_intervened,
                 y_offset=15,
                 **text_options)

        fig.text(text='node', source=ds_node, view=vw_node_intervened,
                 y_offset=-15,  # text_font_style='bold',
                 **text_options)
        fig.text(text='intervention', source=ds_node, view=vw_node_intervened,
                 y_offset=10,
                 **text_options)
        fig.text(text='value', source=ds_node, view=vw_node_intervened,
                 y_offset=25, text_alpha=0.5, text_font_size='12px',
                 **text_options)

        fig.text(text='node', source=ds_node, view=vw_node_predicted,
                 y_offset=-15,
                 **text_options)
        fig.text(text='predictive', source=ds_node, view=vw_node_predicted,
                 y_offset=10, text_font_style='italic',
                 **text_options)
        fig.text(text='value', source=ds_node, view=vw_node_predicted,
                 y_offset=25, text_alpha=0.5, text_font_size='12px',
                 **text_options)

        return fig

    def get_side_layout(self, main_layout, *, width):
        ds_node, ds_table, test_data = self.ds_node, self.ds_table, self.data
        graph = self.bn.graph
        nodes = ds_node.data['node']
        interventions = self.bn.interventions

        def _predictable(node):
            return len(graph.get_parents(node)) > 0 and node not in interventions.keys()

        outcome_nodes = list(filter(_predictable, nodes))

        outcome_title = M.Div(text='<b>Outcome:</b>')
        outcome_widget = M.MultiChoice(
            options=outcome_nodes, max_items=2,
            placeholder='click to select',
            sizing_mode='stretch_width',
        )
        div_scores = M.Div(text='')
        btn = M.Button(label='predict', button_type="primary", disabled=True)
        layout = column(outcome_title, outcome_widget, btn, div_scores,
                        width=width)

        def on_node_change(attr, old_value, new_value):
            if layout.disabled:
                return

            if len(new_value) > 0:
                idx = new_value[0]
                node = nodes[idx]
                if len(outcome_widget.value) == 0 and node in outcome_widget.options:
                    outcome_widget.value = [node]

        def on_value_change(attr, old_value, new_value):
            assert isinstance(new_value, (list, tuple))
            btn.disabled = len(new_value) == 0

        def on_btn_click():
            if len(outcome_widget.value) == 0:
                return

            node_names = outcome_widget.value
            logger.info(f'predict with {node_names}')

            df = test_data.copy()
            y_trues = {}
            pred_names = []
            for c in node_names:
                if c in df.columns.tolist():
                    y_trues[c] = df.pop(c)
            # do predicting
            y_preds = self.bn.predict(df, outcome=node_names)

            for c in node_names:
                pn = self.pred_name(c)
                pred_names.append(pn)

                # update table datasource
                ds_table.data[pn] = y_preds[c].values

                # show scoring
                score_html = '\n'.join(
                    self.format_dict(calc_score(y_true, y_preds=y_preds[c]), title=c)
                    for c, y_true in y_trues.items()
                )
                div_scores.text = score_html

            ppv = self.PPV
            widget_table = main_layout.children[-1].children[-1]
            for c in widget_table.columns:
                if c.field in pred_names:
                    c.visible = True
                elif c.field.startswith(ppv):
                    c.visible = False

        outcome_widget.on_change('value', on_value_change)
        ds_node.selected.on_change('indices', on_node_change)
        btn.on_click(on_btn_click)

        return layout


class BNInterventionView(FittedBNPlotView):
    @property
    def data_title(self):
        return 'Train Data'

    def get_side_layout(self, main_layout, *, width):
        ds_node, ds_table, test_data = self.ds_node, self.ds_table, self.data

        bn = self.bn
        graph = bn.graph
        nodes = ds_node.data['node']
        values = ds_node.data['value']
        intervenable_nodes = list(filter(lambda n: len(graph.get_children(n)) > 0, nodes))

        intervention_applied = M.Div(text=self.intervention_to_html())
        node_choice = M.MultiChoice(
            options=intervenable_nodes, max_items=3,
            placeholder='press to select node',
            sizing_mode='stretch_width',
        )
        intervention_placeholder = M.Div(text='no selected')
        intervention_items = column(intervention_placeholder, sizing_mode='stretch_width')

        btn_do = M.Button(label='do', button_type="primary", disabled=True)
        widgets = [
            M.Div(text='<b>Applied interventions:</b>'),
            intervention_applied,
            M.Div(text='<b>New intervention:</b>'),
            node_choice,
            intervention_items,
            btn_do,
        ]

        intervention_children = {}
        node_selected = []
        intervention_settings = {}

        def on_choice_change(attr, old_value, new_value):
            node_values = ds_node.data['value']
            new_children = []
            for n in new_value:
                if n not in intervention_children.keys():
                    idx = nodes.index(n)
                    widget = self._node_editable_widget(n, node_values[idx], width=width - 10)
                    intervention_children[n] = widget
                    widget.on_change('value', partial(on_intervention_change, n))
                    intervention_settings[n] = widget.value
                new_children.append(intervention_children[n])
            if len(new_children) == 0:
                new_children.append(intervention_placeholder)
            intervention_items.children = new_children

            node_selected.clear()
            node_selected.extend(new_value)
            btn_do.disabled = len(node_selected) == 0

        def on_intervention_change(node, attr, old_value, new_value):
            intervention_settings[node] = new_value

        def on_do():
            assert len(node_selected) > 0
            logger.info(f'do intervention with {node_selected}')

            intervention = {n: intervention_settings[n] for n in node_selected}
            bn.do(intervention, data=test_data, inplace=True)
            intervention_applied.text = self.intervention_to_html()

            if ds_table.selected.indices:
                idx = ds_table.selected.indices[0]
            else:
                idx = 0
            self.patch_node_ds(row_data=idx, params=True, intervention=True)

        node_choice.on_change('value', on_choice_change)
        btn_do.on_click(on_do)

        return column(*widgets, width=width)

    def patch_node_ds(self, *, row_data=None, params=False, intervention=False):
        ds_table = self.ds_table
        ds_node = self.ds_node
        nodes = ds_node.data['node']

        if row_data is not None:
            assert isinstance(row_data, int)
            idx = row_data
            new_data = [self.format_value(n, ds_table.data[n][idx]) for n in nodes]
            ds_node.data['value'] = new_data

        if params:
            if not isinstance(params, dict):
                params = self.get_node_params()
            params_html = [self.params_to_html(params[n]) for n in nodes]
            ds_node.data['params'] = params_html

        if intervention:
            interventions = self.bn.interventions
            intervention_new = [self.format_value(n, interventions.get(n)) for n in nodes]
            ds_node.data['intervention'] = intervention_new
            ds_node.data['line_width'] = [self._line_width(n in interventions.keys()) for n in nodes]

    def intervention_to_html(self):
        return self.format_dict(self.bn.interventions)

    def _node_editable_widget(self, node, value=None, **kwargs):
        state = self.bn.state_[node]
        options = dict(title=f'{node}:')

        if isinstance(state, _base.CategoryNodeState):
            options.update(value=str(value), options=list(map(str, state.classes.tolist())))
            options.update(kwargs)
            widget = M.Select(**options)
        else:
            step = .0001  # state.max - state.max / 20
            fmt = "0[.]0000"
            if value is not None:
                value = float(value)
            options.update(value=value, start=state.min, end=state.max, step=step, format=fmt)
            options.update(kwargs)
            widget = M.Slider(**options)

        return widget


class BNEffectView(FittedBNPlotView):
    def get_side_layout(self, main_layout, *, width):
        ds_node, ds_table, test_data = self.ds_node, self.ds_table, self.data
        bn = self.bn
        graph = bn.graph
        nodes = ds_node.data['node']
        treatment_nodes = list(filter(lambda n: len(graph.get_children(n)) > 0, nodes))
        outcome_nodes = list(filter(lambda n: len(graph.get_parents(n)) > 0, nodes))

        treatment_title = M.Div(text='<b>Treatments:</b>')
        treatment_widget = M.MultiChoice(
            options=treatment_nodes, max_items=2,
            sizing_mode='stretch_width',
        )
        treatment_placeholder = M.Div(text='no selected')
        treatment_items = column(treatment_placeholder, sizing_mode='stretch_width')
        outcome_title = M.Div(text='<hr/><b>Outcome:</b>')
        outcome_widget = M.MultiChoice(
            options=outcome_nodes, max_items=2,
            sizing_mode='stretch_width',
        )
        btn_estimate = M.Button(label='estimate', button_type="primary", disabled=True)
        effect_items = column(M.Div(text='effect'), sizing_mode='stretch_width')

        widgets = [treatment_title, treatment_widget, treatment_items,
                   outcome_title, outcome_widget,
                   btn_estimate, effect_items]
        layout = column(*widgets, width=width)

        treatment_children = {}
        treatment_selected = []
        outcome_selected = []
        treats = {}
        controls = {}

        def on_node_change(attr, old_value, new_value):
            assert isinstance(new_value, (list, tuple))
            if layout.disabled:
                return

            if len(new_value) > 0:
                idx = new_value[0]
                node = ds_node.data['node'][idx]
                if len(treatment_widget.value) == 0 and node in treatment_widget.options:
                    treatment_widget.value = [node]
                elif len(outcome_widget.value) == 0 and node in outcome_widget.options:
                    outcome_widget.value = [node]

        def on_treatment_change(attr, old_value, new_value):
            new_children = []
            for n in new_value:
                if n not in treatment_children.keys():
                    layout, tc = self._node_treatment(n, width=width - 20)
                    treatment_children[n] = (layout, tc)
                    tc[0].on_change('value', partial(on_treat_control_change, n, 0))  # control
                    tc[1].on_change('value', partial(on_treat_control_change, n, 1))  # treat
                    controls[n] = tc[0].value
                    treats[n] = tc[1].value
                new_children.append(treatment_children[n][0])
            if len(new_children) == 0:
                new_children.append(treatment_placeholder)
            treatment_items.children = new_children

            treatment_selected.clear()
            treatment_selected.extend(new_value)
            btn_estimate.disabled = len(treatment_selected) == 0 or len(outcome_widget.value) == 0

        def on_treat_control_change(node, t_or_c, attr, old_value, new_value):
            if t_or_c == 0:
                controls[node] = new_value
            else:
                treats[node] = new_value

        def on_outcome_change(attr, old_value, new_value):
            outcome_selected.clear()
            outcome_selected.extend(new_value)
            btn_estimate.disabled = len(treatment_selected) == 0 or len(outcome_widget.value) == 0

        def on_btn_estimate():
            assert len(treatment_selected) > 0
            logger.info(f'estimate with treatment={treatment_selected}, outcome={outcome_selected}')

            treat, control = nmap(lambda n: (treats[n], controls[n]), treatment_selected)
            df_test = test_data.copy()
            for c in outcome_selected:
                if c in df_test.columns.tolist():
                    df_test.pop(c)

            progress = [0]
            done = []
            ite = []

            def on_progress(n):
                progress[0] = n

            def update_progress():
                ds_progress.data['right'] = progress

                if done:
                    curdoc().remove_periodic_callback(cb)
                    btn_estimate.disabled = False
                    if ite:
                        fig = get_ite_plot(ite[-1], outcome_selected)
                        effect_items.children = [fig]

            def on_success(effect):
                ite.append(effect)
                done.append(1)

            def on_error(e):
                done.append(0)

            n_sample = 200
            fig, ds_progress = get_estimating_plot(n_sample)
            effect_items.children = [fig]

            cb = curdoc().add_periodic_callback(update_progress, 100)
            print(cb)
            btn_estimate.disabled = True

            utils.trun(lambda: bn.estimate(df_test,
                                           outcome=outcome_selected,
                                           treatment=treatment_selected,
                                           treat=treat,
                                           control=control,
                                           num_samples=n_sample,
                                           verbose=on_progress,
                                           random_state=101,
                                           ),
                       on_success=on_success,
                       on_error=on_error,
                       )

        def get_estimating_plot(x_limit):
            source = M.ColumnDataSource(data=dict(y=['progress'], right=[0]))
            fig = figure(title='progress:',
                         toolbar_location=None,
                         outline_line_color='lightgray',
                         x_range=(0, x_limit),
                         y_range=['progress'],
                         height=20,
                         sizing_mode='stretch_width',
                         )
            fig.axis.visible = False
            fig.hbar(y='y', right='right', source=source)
            return fig, source

        def get_ite_plot(ite, outcome):
            from bokeh.palettes import Category10
            fig = figure(toolbar_location=None,
                         # tools="pan,tap",
                         outline_line_color='lightgray',
                         height=width,
                         sizing_mode='stretch_width',
                         )
            ###
            ate = ['ATE:']
            for i, c in enumerate(outcome):
                state = bn.state_[c]
                if isinstance(state, _base.CategoryNodeState):
                    col = f'{c}_{state.classes[-1]}'  # plot the ite for last item
                else:
                    col = c

                x = ite[col].values
                ate.append(f'    {col}: {x.mean():.6f}')
                bins = np.linspace(x.min(), x.max(), 100)
                hist, edges = np.histogram(x, density=True, bins=bins)
                fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                         alpha=0.5, legend_label=col,  # f'{col}(mean {ate:.6f})',
                         fill_color=Category10[3][i],
                         )
                ate_line = M.Span(location=x.mean(), dimension='height',
                                  line_color='#B7472A',
                                  # line_color=Category10[3][i],
                                  line_width=2,
                                  )
                fig.add_layout(ate_line)
            fig.title = '\n'.join(ate)
            fig.legend.visible = len(outcome) > 1
            return fig

        ds_node.selected.on_change('indices', on_node_change)
        treatment_widget.on_change('value', on_treatment_change)
        outcome_widget.on_change('value', on_outcome_change)
        btn_estimate.on_click(on_btn_estimate)

        return layout

    def _node_treatment(self, node, width=None, **kwargs):
        state = self.bn.state_[node]
        options = kwargs.copy()
        options.update(sizing_mode='stretch_width')
        if isinstance(state, _base.CategoryNodeState):
            classes = list(map(str, state.classes.tolist()))
            options.update(options=classes)
            widget_c = M.Select(title='control:', value=classes[0], **options)
            widget_t = M.Select(title='treat:', value=classes[-1], **options)
        else:
            fmt = "0[.]000"
            # step = .001  # state.max - state.max / 20
            # options.update(start=state.min, end=state.max, step=step, format=fmt)
            # widget_c = M.Slider(title='control', value=state.min, **options)
            # widget_t = M.Slider(title='treat', value=state.max, **options)
            options.update(mode='float', low=state.min, high=state.max, format=fmt)
            widget_c = M.NumericInput(title='control:', value=state.min, **options)
            widget_t = M.NumericInput(title='control:', value=state.max, **options)

        layout = column(
            M.Div(text=f'<b>{node}:</b>'),
            row(widget_c, widget_t, sizing_mode='stretch_width'),
            width=width,
            sizing_mode='stretch_width',
        )
        return layout, (widget_c, widget_t)
