import os
import numpy as np
from cxotime import CxoTime
from pprint import pformat
import kadi.commands
import kadi.commands.states as kadi_states
import logging
from Ska.File import get_globfiles

# Define state keys for states, corresponding to the legacy states in
# Chandra.cmd_states.
STATE_KEYS = ['ccd_count', 'clocking', 'dec', 'dither',
              'fep_count', 'hetg', 'letg', 'obsid', 'pcad_mode', 'pitch',
              'power_cmd', 'q1', 'q2', 'q3', 'q4', 'ra', 'roll', 'si_mode',
              'simfa_pos', 'simpos',
              'vid_board']


class StateBuilder(object):
    """
    This is the base class for all StateBuilder objects. It
    should not be used by itself, but subclassed.
    """
    def __init__(self, logger=None):
        if logger is None:
            # Make a logger but with no output
            logger = logging.getLogger('statebuilder-no-logger')
        self.logger = logger

    def get_prediction_states(self, tlm):
        """
        Get the states used for the thermal prediction.

        Parameters
        ----------
        tlm : dictionary
            Dictionary containg temperature and other telemetry
        """
        raise NotImplementedError("'StateBuilder should be subclassed!")

    def get_validation_states(self, datestart, datestop):
        """
        Get states for validation of the thermal model.

        Parameters
        ----------
        datestart : string
            The start date to grab states afterward.
        datestop : string
            The end date to grab states before.
        """
        start = CxoTime(datestart)
        stop = CxoTime(datestop)
        self.logger.info('Getting commanded states between %s - %s' %
                         (start.date, stop.date))

        states = kadi_states.get_states(start, stop, state_keys=STATE_KEYS,
                                        merge_identical=True)

        # Set start and end state date/times to match telemetry span.  Extend the
        # state durations by a small amount because of a precision issue converting
        # to date and back to secs.  (The reference tstop could be just over the
        # 0.001 precision of date and thus cause an out-of-bounds error when
        # interpolating state values).
        dt = 0.01 / 86400
        states['tstart'][0] = (start - dt).secs
        states['datestart'][0] = (start - dt).date
        states['tstop'][-1] = (stop + dt).secs
        states['datestop'][-1] = (stop + dt).date

        return states


class SQLStateBuilder(StateBuilder):
    """
    The SQLStateBuilder contains the original code used to
    obtain commanded states for prediction and validation of
    a thermal model for a particular command load. It can also
    be used for validation only.

    SQL is no longer relevant as the commands and states come from kadi.
    """
    def __init__(self, interrupt=False, backstop_file=None,
                 logger=None):
        """
        Give the SQLStateBuilder arguments that were passed in
        from the command line, and set up the connection to the
        commanded states database.

        Parameters
        ----------
        interrupt : boolean
            If True, this is an interrupt load.
        backstop_file : string
            Path to the backstop file. If a directory, the backstop
            file will be searched for within this directory.
        logger : Logger object, optional
            The Python Logger object to be used when logging.
        """
        super(SQLStateBuilder, self).__init__(logger=logger)

        # Note: `interrupt` is ignored in this class. This concept is not needed
        # since backstop 6.9, which provides the RUNNING_LOAD_TERMINATION_TIME
        # (RLTT) that can be reliably used to remove scheduled commands that
        # will not be run.
        self.interrupt = interrupt
        self.backstop_file = backstop_file
        self._get_bs_cmds()

    def _get_bs_cmds(self):
        """
        Internal method used to obtain commands from the backstop
        file and store them.
        """
        if os.path.isdir(self.backstop_file):
            # Returns a list but requires exactly 1 match
            backstop_file = get_globfiles(os.path.join(self.backstop_file,
                                                       'CR*.backstop'))[0]
            self.backstop_file = backstop_file

        self.logger.info('Using backstop file %s' % self.backstop_file)

        # Read the backstop commands and add a `time` column
        bs_cmds = kadi.commands.get_cmds_from_backstop(self.backstop_file)
        bs_cmds['time'] = CxoTime(bs_cmds['date']).secs

        self.bs_cmds = bs_cmds
        self.tstart = bs_cmds[0]['time']
        self.tstop = bs_cmds[-1]['time']

    def get_prediction_states(self, tbegin):
        """
        Get the states used for the prediction.

        Parameters
        ----------
        tbegin : string
            The starting date/time from which to obtain states for
            prediction.
        """
        # This is a kadi.commands.CommandTable (subclass of astropy Table)
        bs_cmds = self.bs_cmds
        bs_dates = bs_cmds['date']

        # Running loads termination time is the last time of "current running
        # loads" (or in the case of a safing action, "current approved load
        # commands" in kadi commands) which should be included in propagation.
        # Starting from around 2020-April (backstop 6.9) this is included as a
        # commmand in the loads, while prior to that we just use the first
        # command in the backstop loads.
        ok = bs_cmds['event_type'] == 'RUNNING_LOAD_TERMINATION_TIME'
        if np.any(ok):
            rltt = CxoTime(bs_dates[ok][0])
        else:
            # Handle the case of old loads (prior to backstop 6.9) where there
            # is no RLTT.  If the first command is AOACRSTD this indicates the
            # beginning of a maneuver ATS which may overlap by 3 mins with the
            # previous loads because of the AOACRSTD command. So move the RLTT
            # forward by 3 minutes (exactly 180.0 sec). If the first command is
            # not AOACRSTD then that command time is used as RLTT.
            if bs_cmds['tlmsid'][0] == 'AOACRSTD':
                rltt = CxoTime(bs_cmds['time'][0] + 180)
            else:
                rltt = CxoTime(bs_cmds['date'][0])

        # Scheduled stop time is the end of propagation, either the explicit
        # time as a pseudo-command in the loads or the last backstop command time.
        ok = bs_cmds['event_type'] == 'SCHEDULED_STOP_TIME'
        sched_stop = CxoTime(bs_dates[ok][0] if np.any(ok) else bs_dates[-1])

        self.logger.info(f'RLTT = {rltt.date}')
        self.logger.info(f'sched_stop = {sched_stop.date}')

        # Get currently running (or approved) commands from tbegin up to and
        # including commands at RLTT
        cmds = kadi.commands.get_cmds(tbegin, rltt, inclusive_stop=True)

        # Add in the backstop commands
        cmds = cmds.add_cmds(bs_cmds)

        # Get the states for available commands, boxed by tbegin / sched_stop.
        # The merge_identical=False is for compatibility with legacy Chandra.cmd_states,
        # but this could probably be set to True.
        states = kadi_states.get_states(cmds=cmds, start=tbegin, stop=sched_stop,
                                        state_keys=STATE_KEYS,
                                        merge_identical=False)

        # Make the column order match legacy Chandra.cmd_states.
        states = states[sorted(states.colnames)]

        # Get the first state as a dict.
        state0 = {key: states[0][key] for key in states.colnames}

        return states, state0


#-------------------------------------------------------------------------------
# ACIS Ops Load History assembly
#-------------------------------------------------------------------------------
class ACISStateBuilder(StateBuilder):

    def __init__(self, interrupt=False, backstop_file=None, nlet_file=None,
                 logger=None):
        """
        Give the ACISStateBuilder arguments that were passed in
        from the command line and get the backstop commands from the load
        under review.

        Parameters
        ----------
        interrupt : boolean
            If True, this is an interrupt load.
        backstop_file : string
            Path to the backstop file. If a directory, the backstop
            file will be searched for within this directory.
        nlet_file : string
            full path to the Non-Load Event Tracking file
        logger : Logger object, optional
            The Python Logger object to be used when logging.
        """
        # Import the BackstopHistory class
        from backstop_history import BackstopHistory

        # Capture the full path to the NLET file to be used
        self.nlet_file = nlet_file

        # Create an instance of the Backstop Command class
        self.BSC = BackstopHistory.BackstopHistory('ACIS-Continuity.txt', self.nlet_file)

        # The Review Load backstop name
        self.rev_bs_name = None
        # Normally I would have created the self.rev_bs_cmds attribute
        # and used that however to work with ATC I changed it to bs_cmds.

        super(ACISStateBuilder, self).__init__(logger=logger)
        self.interrupt = interrupt
        self.backstop_file = backstop_file

        # if the user supplied a full path to the backstop file then
        # capture the backstop file name and the commands within the backstop file.
        if backstop_file is not None:
            # Get tstart, tstop, commands from backstop file in args.oflsdir
            # These are the REVIEW backstop commands. This returns a list of dict
            # representing the commands.
            rev_bs_cmds, self.rev_bs_name = self.BSC.get_bs_cmds(self.backstop_file)

            # Store the Review Load backstop commands in the class attribute and
            # also capture the Review load time of first command (TOFC) and
            # Time of Last Command (TOLC).
            self.bs_cmds = rev_bs_cmds
            self.tstart = rev_bs_cmds[0]['time']
            self.tstop = rev_bs_cmds[-1]['time']

            # Initialize the end time attribute for event searches within the BSC object
            # At the beginning, it will be the time of the last command in the Review Load
            self.BSC.end_event_time = rev_bs_cmds[-1]['time']

    def get_prediction_states(self, tbegin):
        """
        Get the states used for the prediction.  This includes both the
        states from the review load backstop file and all the
        states between the latest telemetry data and the beginning
        of that review load backstop file.

        The Review Backstop commands already obtained.
        Telemtry from 21 days back to the latest in Ska obtained.

        So now the task is to backchain through the loads and assemble
        any states missing between the end of telemetry through the start
        of the review load.

        Parameters
        ----------
        tbegin : string
            The starting date/time from which to obtain states for
            prediction. This is tlm['date'][-5]) or, in other words, the
            date used is approximately 30 minutes before the end of the
            fetched telemetry
        """
        # If an OFLS directory has been specified, get the backstop commands
        # stored in the backstop file in that directory

        # Ok ready to start the collection of continuity commands
        #
        # Make a copy of the Review Load Commands. This will have
        # Continuity commands concatenated to it and will be the final product

        import copy

        # List of dict representing commands at this point
        bs_cmds = copy.copy(self.bs_cmds)

        # Capture the start time of the review load
        bs_start_time = bs_cmds[0]['time']

        # Capture the path to the ofls directory
        present_ofls_dir = copy.copy(self.backstop_file)

        # So long as the earliest command in bs_cmds is after the state0 time
        # (which is the same as tbegin), keep concatenating continuity commands
        # to bs_cmds based upon the type of load. Note that as you march back in
        # time along the load chain, "present_ofls_dir" will change.

        # WHILE
        # The big while loop that backchains through previous loads and concatenates the
        # proper load sections to the review load.
        while CxoTime(tbegin).secs < bs_start_time:

            # Read the Continuity information of the present ofls directory
            cont_load_path, present_load_type, scs107_date = self.BSC.get_continuity_file_info(present_ofls_dir)

            #---------------------- NORMAL ----------------------------------------
            # If the load type is "normal" then grab the continuity command
            # set and concatenate those commands to the start of bs_cmds
            if present_load_type.upper() == 'NORMAL':
                # Obtain the continuity load commands
                cont_bs_cmds, cont_bs_name = self.BSC.get_bs_cmds(cont_load_path)

                # Combine the continuity commands with the bs_cmds. The result
                # is stored in bs_cmds
                bs_cmds = self.BSC.CombineNormal(cont_bs_cmds, bs_cmds)

                # Reset the backstop collection start time for the While loop
                bs_start_time = bs_cmds[0]['time']
                # Now point the operative ofls directory to the Continuity directory
                present_ofls_dir = cont_load_path

            #---------------------- TOO ----------------------------------------
            # If the load type is "too" then grab the continuity command
            # set and concatenate those commands to the start of bs_cmds
            elif present_load_type.upper() == 'TOO':
                # Obtain the continuity load commands
                cont_bs_cmds, cont_bs_name = self.BSC.get_bs_cmds(cont_load_path)

                # Combine the continuity commands with the bs_cmds
                bs_cmds = self.BSC.CombineTOO(cont_bs_cmds, bs_cmds)

                # Reset the backstop collection start time for the While loop
                bs_start_time = bs_cmds[0]['time']
                # Now point the operative ofls directory to the Continuity directory
                present_ofls_dir = cont_load_path

            #---------------------- STOP ----------------------------------------
            # If the load type is "STOP" then grab the continuity command
            # set and concatenate those commands to the start of bs_cmds
            # Take into account the SCS-107 commands which shut ACIS down
            # and any LTCTI run
            elif present_load_type.upper() == 'STOP':

                # Obtain the continuity load commands
                cont_bs_cmds, cont_bs_name = self.BSC.get_bs_cmds(cont_load_path)

                # CombineSTOP the continuity commands with the bs_cmds
                bs_cmds = self.BSC.CombineSTOP(cont_bs_cmds, bs_cmds, scs107_date )

                # Reset the backstop collection start time for the While loop
                bs_start_time = bs_cmds[0]['time']
                # Now point the operative ofls directory to the Continuity directory
                present_ofls_dir = cont_load_path

            #---------------------- SCS-107 ----------------------------------------
            # If the load type is "STOP" then grab the continuity command
            # set and concatenate those commands to the start of bs_cmds
            # Take into account the SCS-107 commands which shut ACIS down
            # and any LTCTI run
            elif present_load_type.upper() == 'SCS-107':
                # Obtain the continuity load commands
                cont_bs_cmds, cont_bs_name = self.BSC.get_bs_cmds(cont_load_path)
                # Store the continuity bs commands as a chunk in the chunk list

                # Obtain the CONTINUITY load Vehicle-Only file
                vo_bs_cmds, vo_bs_name = self.BSC.get_vehicle_only_bs_cmds(cont_load_path)

                # Combine107 the continuity commands with the bs_cmds
                bs_cmds = self.BSC.Combine107(cont_bs_cmds, vo_bs_cmds, bs_cmds, scs107_date)

                # Reset the backstop collection start time for the While loop
                bs_start_time = bs_cmds[0]['time']
                # Now point the operative ofls directory to the Continuity directory
                present_ofls_dir = cont_load_path

        # Convert backstop commands from a list of dict to a CommandTable and
        # store in self.
        bs_cmds = kadi.commands.CommandTable(bs_cmds)
        self.bs_cmds = bs_cmds

        # Clip commands to tbegin
        bs_cmds = bs_cmds[bs_cmds['date'] > tbegin]

        # Scheduled stop time is the end of propagation, either the explicit
        # time as a pseudo-command in the loads or the last backstop command time.
        # Use the last such command if any are found (which is always the case
        # since backstop 6.9).
        ok = bs_cmds['event_type'] == 'SCHEDULED_STOP_TIME'
        sched_stop = bs_cmds['date'][ok][-1] if np.any(ok) else bs_cmds['date'][-1]

        # Convert the assembled command history into commanded states
        # corresponding to the commands. This includes continuity commanding
        # from the end of telemetry along with the in-review load backstop
        # commands.
        states = kadi_states.get_states(cmds=bs_cmds, start=tbegin, stop=sched_stop,
                                        state_keys=STATE_KEYS)

        # Make the column order match legacy Chandra.cmd_states.
        states = states[sorted(states.colnames)]

        # Get the first state as a dict.
        state0 = {key: states[0][key] for key in states.colnames}

        self.logger.debug(f"state0 at {CxoTime(state0['tstart']).date} "
                          f"is\n{pformat(state0)}")

        return states, state0


state_builders = {"sql": SQLStateBuilder,
                  "acis": ACISStateBuilder}
