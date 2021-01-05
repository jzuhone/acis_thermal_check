import numpy as np
import Ska.Sun
import logging
import os
import matplotlib.pyplot as plt
from Ska.Matplotlib import cxctime2plotdate
import Ska.Numpy

TASK_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

mylog = logging.getLogger('acis_thermal_check')

thermal_blue = 'blue'
thermal_red = 'red'


def calc_pitch_roll(times, ephem, states):
    """Calculate the normalized sun vector in body coordinates.
    Shamelessly copied from Ska.engarchive.derived.pcad but 
    modified to use commanded states quaternions

    Parameters
    ----------
    times : NumPy array of times in seconds
    ephem : orbitephem and solarephem info 
    states : commanded states NumPy recarray

    Returns
    -------
    3 NumPy arrays: time, pitch and roll
    """
    from Ska.engarchive.derived.pcad import arccos_clip, qrotate
    idxs = Ska.Numpy.interpolate(np.arange(len(states)), states['tstart'],
                                 times, method='nearest')
    states = states[idxs]

    chandra_eci = np.array([ephem['orbitephem0_x'],
                            ephem['orbitephem0_y'],
                            ephem['orbitephem0_z']])
    sun_eci = np.array([ephem['solarephem0_x'],
                        ephem['solarephem0_y'],
                        ephem['solarephem0_z']])
    sun_vec = -chandra_eci + sun_eci
    est_quat = np.array([states['q1'],
                         states['q2'],
                         states['q3'],
                         states['q4']])

    sun_vec_b = qrotate(est_quat, sun_vec)  # Rotate into body frame
    magnitude = np.sqrt((sun_vec_b ** 2).sum(axis=0))
    magnitude[magnitude == 0.0] = 1.0
    sun_vec_b = sun_vec_b / magnitude  # Normalize

    pitch = np.degrees(arccos_clip(sun_vec_b[0, :]))
    roll = np.degrees(np.arctan2(-sun_vec_b[1, :], -sun_vec_b[2, :]))

    return pitch, roll


def config_logging(outdir, verbose):
    """
    Set up file and console logger.
    See http://docs.python.org/library/logging.html#logging-to-multiple-destinations
    Logs to the console and to run.dat.

    Parameters
    ----------
    outdir : string
        The location of the directory which the model outputs
        are being written to.
    verbose : integer
        Indicate how verbose we want the logger to be.
        (0=quiet, 1=normal, 2=debug)
    """
    # Disable auto-configuration of root logger by adding a null handler.
    # This prevents other modules (e.g. Chandra.cmd_states) from generating
    # a streamhandler by just calling logging.info(..).
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
    rootlogger = logging.getLogger()
    rootlogger.addHandler(NullHandler())

    logger = logging.getLogger('acis_thermal_check')
    logger.setLevel(logging.DEBUG)

    # Set numerical values for the different log levels
    loglevel = {0: logging.CRITICAL,
                1: logging.INFO,
                2: logging.DEBUG}.get(verbose, logging.INFO)

    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(loglevel)
    logger.addHandler(console)

    logfile = os.path.join(outdir, 'run.dat')

    filehandler = logging.FileHandler(filename=logfile, mode='w')
    filehandler.setFormatter(formatter)
    # Set the file loglevel to be at least INFO,
    # but override to DEBUG if that is requested at the
    # command line
    filehandler.setLevel(logging.INFO)
    if loglevel == logging.DEBUG:
        filehandler.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)


def plot_one(fig_id, x, y, yy=None, linestyle='-',
             ll='--', color=thermal_blue, 
             linewidth=2, xmin=None, xmax=None, 
             ylim=None, xlabel='', ylabel='', title='',
             figsize=(12, 6), load_start=None,
             width=None):
    """
    Plot one quantities with a date x-axis and a left
    y-axis.

    Parameters
    ----------
    fig_id : integer
        The ID for this particular figure.
    x : NumPy array
        Times in seconds since the beginning of the mission for
        the left y-axis quantity.
    y : NumPy array
        Quantity to plot against the times on the left x-axis.
    yy : NumPy array, optional
        A second quantity to plot against the times on the 
        left x-axis. Default: None
    linestyle : string, optional
        The style of the line for the left y-axis.
    ll : string, optional
        The style of the second line for the left y-axis.
    color : string, optional
        The color of the line for the left y-axis.
    linewidth : string, optional
        The width of the lines. Default: 2
    xmin : float, optional
        The left-most value of the x-axis.
    xmax : float, optional
        The right-most value of the x-axis.
    ylim : 2-tuple, optional
        The limits for the left y-axis.
    xlabel : string, optional
        The label of the x-axis.
    ylabel : string, optional
        The label for the left y-axis.
    title : string, optional
        The title for the plot.
    figsize : 2-tuple of floats
        Size of plot in width and height in inches.
    """
    # Convert times to dates
    xt = cxctime2plotdate(x)
    fig = plt.figure(fig_id, figsize=figsize)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    # Plot left y-axis
    ax.plot_date(xt, y, fmt='-', linestyle=linestyle, linewidth=linewidth, 
                 color=color)
    if yy is not None:
        ax.plot_date(xt, yy, fmt='-', linestyle=ll, linewidth=linewidth, 
                     color=color)
    if xmin is None:
        xmin = min(xt)
    if xmax is None:
        xmax = max(xt)
    ax.set_xlim(xmin, xmax)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()

    if load_start is not None:
        # Add a vertical line to mark the start time of the load
        ax.axvline(load_start, linestyle='-', color='g', linewidth=2.0)

    Ska.Matplotlib.set_time_ticks(ax)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation_mode("anchor")
        label.set_rotation(30)
        label.set_horizontalalignment('right')
    ax.tick_params(which='major', axis='x', length=6)
    ax.tick_params(which='minor', axis='x', length=3)

    fig.subplots_adjust(bottom=0.22, right=0.87)
    # The next several lines ensure that the width of the axes
    # of all the weekly prediction plots are the same
    if width is not None:
        w2, _ = fig.get_size_inches()
        lm = fig.subplotpars.left * width / w2
        rm = fig.subplotpars.right * width / w2
        fig.subplots_adjust(left=lm, right=rm)

    return {'fig': fig, 'ax': ax}


def plot_two(fig_id, x, y, x2, y2, yy=None, linewidth=2,
             linestyle='-', linestyle2='-', ll='--', 
             color=thermal_blue, color2='magenta',
             xmin=None, xmax=None, ylim=None, ylim2=None,
             xlabel='', ylabel='', ylabel2='', title='',
             figsize=(12, 6), load_start=None, width=None):
    """
    Plot two quantities with a date x-axis, one on the left
    y-axis and the other on the right y-axis.

    Parameters
    ----------
    fig_id : integer
        The ID for this particular figure.
    x : NumPy array
        Times in seconds since the beginning of the mission for
        the left y-axis quantity.
    y : NumPy array
        Quantity to plot against the times on the left x-axis.
    x2 : NumPy array
        Times in seconds since the beginning of the mission for
        the right y-axis quantity.
    y2 : NumPy array
        Quantity to plot against the times on the right y-axis.
    yy : NumPy array, optional
        A second quantity to plot against the times on the 
        left x-axis. Default: None
    linewidth : string, optional
        The width of the lines. Default: 2
    linestyle : string, optional
        The style of the line for the left y-axis.
    linestyle2 : string, optional
        The style of the line for the right y-axis.
    ll : string, optional
        The style of the second line for the left y-axis.
    color : string, optional
        The color of the line for the left y-axis.
    color2 : string, optional
        The color of the line for the right y-axis.
    xmin : float, optional
        The left-most value of the x-axis.
    xmax : float, optional
        The right-most value of the x-axis.
    ylim : 2-tuple, optional
        The limits for the left y-axis.
    ylim2 : 2-tuple, optional
        The limits for the right y-axis.
    xlabel : string, optional
        The label of the x-axis.
    ylabel : string, optional
        The label for the left y-axis.
    ylabel2 : string, optional
        The label for the right y-axis.
    title : string, optional
        The title for the plot.
    figsize : 2-tuple of floats
        Size of plot in width and height in inches.
    """
    # Convert times to dates
    xt = cxctime2plotdate(x)
    fig = plt.figure(fig_id, figsize=figsize)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    # Plot left y-axis
    ax.plot_date(xt, y, fmt='-', linestyle=linestyle, linewidth=linewidth,
                 color=color)
    if yy is not None:
        ax.plot_date(xt, yy, fmt='-', linestyle=ll, linewidth=linewidth,
                     color=color)
    if xmin is None:
        xmin = min(xt)
    if xmax is None:
        xmax = max(xt)
    ax.set_xlim(xmin, xmax)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()

    # Plot right y-axis

    ax2 = ax.twinx()
    xt2 = cxctime2plotdate(x2)
    ax2.plot_date(xt2, y2, fmt='-', linestyle=linestyle2, linewidth=linewidth,
                  color=color2)
    ax2.set_xlim(xmin, xmax)
    if ylim2:
        ax2.set_ylim(*ylim2)
    ax2.set_ylabel(ylabel2, color=color2)
    ax2.xaxis.set_visible(False)

    if load_start is not None:
        # Add a vertical line to mark the start time of the load
        ax.axvline(load_start, linestyle='-', color='g', linewidth=2.0)

    Ska.Matplotlib.set_time_ticks(ax)
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation_mode("anchor")
        label.set_rotation(30)
        label.set_horizontalalignment('right')
    [label.set_color(color2) for label in ax2.yaxis.get_ticklabels()]
    ax.tick_params(which='major', axis='x', length=6)
    ax.tick_params(which='minor', axis='x', length=3)
    fig.subplots_adjust(bottom=0.22, right=0.87)
    # The next several lines ensure that the width of the axes
    # of all the weekly prediction plots are the same
    if width is not None:
        w2, _ = fig.get_size_inches()
        lm = fig.subplotpars.left * width / w2
        rm = fig.subplotpars.right * width / w2
        fig.subplots_adjust(left=lm, right=rm)

    ax.set_zorder(10)
    ax.patch.set_visible(False)

    return {'fig': fig, 'ax': ax, 'ax2': ax2}


def get_options(name, model_path, opts=None):
    """
    Construct the argument parser for command-line options for running
    predictions and validations for a load. Sets up the parser and 
    defines default options. This function should be used by the specific 
    thermal model checking tools.

    Parameters
    ----------
    name : string
        The name of the ACIS component whose temperature is being modeled.
    model_path : string
        The default directory path where the model JSON files are located.
        This is internal to the ``acis_thermal_check`` package.
    opts: dictionary
        A (key, value) dictionary of additional options for the parser. These
        may be defined by the thermal model checking tool if necessary.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.set_defaults()
    parser.add_argument("--outdir", default="out", help="Output directory. If it does not "
                                                        "exist it will be created. Default: 'out'")
    parser.add_argument("--backstop_file", help="Path to the backstop file. If a directory, "
                                                "the backstop file will be searched for within "
                                                "this directory. Default: None")
    parser.add_argument("--oflsdir", help="Path to the directory containing the backstop "
                                          "file (legacy argument). If specified, it will "
                                          "override the value of the backstop_file "
                                          "argument. Default: None")
    parser.add_argument("--model-spec", 
                        default=os.path.join(model_path, '%s_model_spec.json' % name),
                        help="Model specification file. Defaults to the one included with "
                             "the model package.")
    parser.add_argument("--days", type=float, default=21.0,
                        help="Days of validation data. Default: 21")
    parser.add_argument("--run-start", help="Reference time to replace run start time "
                                            "for regression testing. The default is to "
                                            "use the current time.")
    parser.add_argument("--interrupt", help="Set this flag if this is an interrupt load.",
                        action='store_true')
    parser.add_argument("--traceback", action='store_false', help='Enable tracebacks. Default: True')
    parser.add_argument("--pred-only", action='store_true', help='Only make predictions. Default: False')
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity (0=quiet, 1=normal, 2=debug)")
    parser.add_argument("--T-init", type=float,
                        help="Starting temperature (degC). Default is to compute it from telemetry.")
    parser.add_argument("--state-builder", default="acis",
                        help="StateBuilder to use (sql|acis). Default: acis")
    parser.add_argument("--nlet_file",
                        default='/data/acis/LoadReviews/NonLoadTrackedEvents.txt',
                        help="Full path to the Non-Load Event Tracking file that should be "
                             "used for this model run.")
    parser.add_argument("--version", action='store_true', help="Print version")

    if opts is not None:
        for opt_name, opt in opts:
            parser.add_argument("--%s" % opt_name, **opt)

    args = parser.parse_args()

    if args.oflsdir is not None:
        args.backstop_file = args.oflsdir

    if args.pred_only and args.backstop_file is None:
        raise RuntimeError("You turned off both prediction and validation!!")

    return args


def make_state_builder(name, args):
    """
    Take the command-line arguments and use them to construct
    a StateBuilder object which will be used for the thermal
    prediction and validation.

    Parameters
    ----------
    name : string 
        The identifier for the state builder to be used.
    args : ArgumentParser arguments
        The arguments to pass to the StateBuilder subclass.
    """
    # Import the dictionary of possible state builders. This
    # dictionary is located in state_builder.py
    from acis_thermal_check.state_builder import state_builders

    builder_class = state_builders[name]

    # Build the appropriate state_builder depending upon the
    # value of the passed in parameter "name" which was
    # originally the --state-builder="sql"|"acis" input argument
    #
    # Instantiate the SQL History Builder: SQLStateBuilder
    if name == "sql":
        state_builder = builder_class(interrupt=args.interrupt,
                                      backstop_file=args.backstop_file,
                                      logger=mylog)

    # Instantiate the ACIS OPS History Builder: ACISStateBuilder
    elif name == "acis":
        # Create a state builder using the ACIS Ops backstop history
        # modules
        state_builder = builder_class(interrupt=args.interrupt,
                                      backstop_file=args.backstop_file,
                                      nlet_file=args.nlet_file,
                                      logger=mylog)
    else:
        raise RuntimeError("No such state builder with name %s!" % name)

    return state_builder


def get_acis_limits(msid):
    """
    Get the current yellow hi limit and margin for a 
    given ACIS-related MSID, or the various limits 
    for the focal plane temperature.

    Parameters
    ----------
    msid : string
        The MSID to get the limits for, e.g. "1deamzt".
    """
    import requests

    if msid == "fptemp":
        cold_ecs = -119.5 # the limit for cold ECS measurements in the science orbit
        acis_i = -112.0 # the limit for ACIS-I observations
        acis_s = -111.0 # the limit for normal ACIS-S observations
        acis_hot = -109.0 # the limit for ACIS-S observations which can go hotter
        return cold_ecs, acis_i, acis_s, acis_hot

    yellow_lo = None
    yellow_hi = None

    margins = {"1dpamzt": 2.0,
               "1deamzt": 2.0,
               "1pdeaat": 4.5,
               "tmp_fep1_mong": 2.0,
               "tmp_fep1_actel": 2.0,
               "tmp_bep_pcb": 2.0}

    margin = margins[msid]

    pmon_file = "PMON/pmon_limits.txt"
    eng_file = "Thermal/MSID_Limits.txt"
    file_root = "/proj/web-cxc/htdocs/acis/"

    if msid.startswith("tmp_"):
        limits_file = pmon_file
        cols = (4, 5)
        msid = "ADC_"+msid.upper()
    else:
        limits_file = eng_file
        cols = (2, 3)

    if os.path.exists(file_root):
        loc = "local"
        f = open(os.path.join(file_root, limits_file), "r")
        lines = f.readlines()
        f.close()
    else:
        loc = "remote"
        url = "http://cxc.cfa.harvard.edu/acis/{}".format(limits_file)
        u = requests.get(url)
        lines = u.text.split("\n")

    mylog.info("Obtaining limits for %s from %s file." % (msid, loc))

    for line in lines:
        words = line.strip().split()
        if len(words) > 1 and words[0] == msid.upper():
            yellow_lo = float(words[cols[0]])
            yellow_hi = float(words[cols[1]])
            break

    return yellow_lo, yellow_hi, margin


def paint_perigee(perigee_passages, states, plots):
    """
    This function draws vertical dashed lines for EEF, Perigee and XEF
    events in the load.EEF and XEF lines are black; Perigee is red.

    You supply the list of perigee passage events which are:
        Radzone Start/Stop time
        Perigee Passage time

        The states you created in main

        The dictionary of plots you created

        The MSID (in this case FP_TEMP) used to access the dictionary
    """
    #
    # Now plot any perigee passages that occur between xmin and xmax
    from cxotime import CxoTime
    for plot in plots.values():
        for eachpassage in perigee_passages:
            # The index [1] item is always the Perigee Passage time. Draw that
            # line in red If this line is between tstart and tstop then it
            # needs to be drawn on the plot. otherwise ignore
            if states['tstop'][-1] >= CxoTime(eachpassage[0]).secs >= states['tstart'][0]:
                # Have to convert this time into the new x axis time scale
                # necessitated by SKA
                xpos = cxctime2plotdate([CxoTime(eachpassage[0]).secs])

                ymin, ymax = plot['ax'].get_ylim()

                # now plot the line.
                plot['ax'].vlines(xpos, ymin, ymax, linestyle=':', color='red',
                                  linewidth=2.0)

                # Plot the perigee passage time so long as it was specified in
                # the CTI_report file
                if eachpassage[1] != "Not-within-load":
                    perigee_time = cxctime2plotdate([CxoTime(eachpassage[1]).secs])
                    plot['ax'].vlines(perigee_time, ymin, ymax, linestyle=':',
                                      color='black', linewidth=2.0)
