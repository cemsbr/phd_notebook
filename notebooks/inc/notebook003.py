from math import ceil

from subprocess import run, PIPE
from sparklogstats.logparser import LogParser


BLOCK_SIZE = 128 * 1024**2


def get_logs():
    process = run('find ../data -name "app-*"', shell=True, check=True,
                  stdout=PIPE)
    return process.stdout.decode().split('\n')[:-1]


def nr_tasks(app):
    return sum(1 for _ in app.stages[0].successful_tasks)


def nr_blocks(app):
    return ceil(app.bytes_read / BLOCK_SIZE)

parser = LogParser()


def parse_log(filename):
    return parser.parse_file(filename)


def analyze_all_logs():
    equal, different, max_diff = 0, 0, 0
    max_app = None
    for log in get_logs():
        app = parse_log(log)
        tasks = nr_tasks(app)
        blocks = nr_blocks(app)
        if blocks != tasks:
            msg = '{} has {} blocks, but {} successful tasks.'
            filename = app.filename[8:]
            print(msg.format(filename, blocks, tasks))
            different += 1
            diff = abs(blocks - tasks)
            if diff > max_diff:
                max_diff = diff
                max_app = app
        else:
            equal += 1
    print('{:d} matches, {:d} differences.'.format(equal, different))
    msg = 'Biggest difference is {}, in {}: {} blocks, {} tasks.'
    print(msg.format(max_diff, max_app.filename, nr_blocks(max_app),
                     nr_tasks(max_app)))
