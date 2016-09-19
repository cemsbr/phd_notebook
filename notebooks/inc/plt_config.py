"""Configure plots."""
import matplotlib


matplotlib.style.use('ggplot')  # R style
matplotlib.rcParams.update({'font.size': 10,
                            'font.family': 'serif',
                            'lines.linewidth': 2,
                            'axes.xmargin': 0.1,
                            'axes.ymargin': 0.1})
