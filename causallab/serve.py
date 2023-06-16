import argparse
import os
import sys

from bokeh.command.subcommands.serve import Serve

from ylearn import __version__


class MyServe(Serve):
    work_dir = None

    def customize_applications(self, args, applications):
        apps = super().customize_applications(args, applications)
        if isinstance(apps, dict) and len(apps) == 1:
            apps = {'/lab': next(iter(apps.values()))}
        return apps

    def customize_kwargs(self, args, server_kwargs):
        kwargs = super().customize_kwargs(args, server_kwargs)

        if self.work_dir is not None:
            from tornado.web import StaticFileHandler
            my_handlers = [
                (r'/download/(.*)', StaticFileHandler, dict(path=self.work_dir)),
            ]
            extra_patterns = kwargs.get('extra_patterns', [])
            extra_patterns.extend(my_handlers)
            kwargs['extra_patterns'] = extra_patterns

        return kwargs


def init_argparser(parser):
    parser.add_argument('--data', '-D', type=str, required=False,
                        help='data file')
    parser.add_argument('--test', '-T', type=str, required=False,
                        help='test data file, optional')
    parser.add_argument('--experiment', '-X', type=str, required=False,
                        help='experiment file')
    parser.add_argument('--work-dir', '-W', type=str, default='~/.causallab/tmp',
                        help='a directory for storing temporary files')


def run_cleaner(path, file_pattern='*.tmp', interval=60, keep_duration=3600):
    from threading import Thread
    import time
    import glob
    import os
    from ylearn.utils import logging
    logger = logging.getLogger('run_cleaner')

    def clean():
        last_at = 0
        while True:
            now = time.time()
            to_sleep = min(interval - abs(now - last_at), 1.0)
            if to_sleep > 0:
                time.sleep(to_sleep)
                continue

            if not os.path.exists(path):
                last_at = time.time()
                continue

            for f in glob.glob(os.path.join(path, file_pattern), recursive=False):
                assert f.startswith(path)
                if os.path.getmtime(f) + keep_duration < now:
                    try:
                        os.remove(f)
                        logger.info(f'{f} removed')
                    except:
                        logger.warm(f'failed to remove file {f}')
            last_at = time.time()

    t = Thread(target=clean, daemon=True)
    t.start()


def main(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        epilog='')

    parser.add_argument('-v', '--version', action='version', version=__version__)
    init_argparser(parser)

    arg_parsed, argv = parser.parse_known_args()
    # if arg_parsed.data is None and arg_parsed.test is None:
    #     raise ValueError('--data or --test is required.')

    fixed_argv = sys.argv[1:] + [__file__, ] + [
        '--args',
    ]
    if arg_parsed.data:
        fixed_argv.extend(['--data', arg_parsed.data])
    if arg_parsed.test:
        fixed_argv.extend(['--test', arg_parsed.test])
    if arg_parsed.experiment:
        fixed_argv.extend(['--experiment', arg_parsed.experiment])
    if arg_parsed.work_dir:
        fixed_argv.extend(['--work-dir', arg_parsed.work_dir])
    # print('fixed_argv:', fixed_argv)

    if arg_parsed.work_dir:
        work_dir = os.path.expanduser(arg_parsed.work_dir)
        os.makedirs(work_dir, exist_ok=True)
        arg_parsed.work_dir = work_dir

    serve = MyServe(parser=parser)
    serve.work_dir = arg_parsed.work_dir
    parser.set_defaults(invoke=serve.invoke)
    args = parser.parse_args(fixed_argv)

    try:
        if arg_parsed.work_dir:
            run_cleaner(path=arg_parsed.work_dir,
                        file_pattern='experiment_*.pkl.gz'
                        )
        ret = args.invoke(args)
    except Exception as e:
        print("ERROR: " + str(e), file=sys.stderr)
        exit(1)

    if ret is False:
        sys.exit(1)
    elif ret is not True and isinstance(ret, int) and ret != 0:
        sys.exit(ret)


def bkapp():
    from causallab.plot import BNExperimentPlotter

    parser = argparse.ArgumentParser('')
    init_argparser(parser)
    args = parser.parse_args()
    work_dir = args.work_dir
    if work_dir:
        work_dir = os.path.expanduser(work_dir)
    plotter = BNExperimentPlotter(
        experiment_file=args.experiment,
        data_file=args.data,
        test_file=args.test,
        work_dir=work_dir,
    )
    plotter.plot()


# print('__name__: >>', __name__)
# print('argv', sys.argv)

if __name__ == '__main__':
    main(sys.argv)
elif __name__.startswith('bokeh_app_'):
    bkapp()
