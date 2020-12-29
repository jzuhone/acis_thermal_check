import os
from numpy.testing import assert_array_equal, \
    assert_allclose
import shutil
import numpy as np
import tempfile
import pickle
from pathlib import Path

months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Loads for regression testing
test_loads = {"normal": ["MAR0617A", "MAR2017E", "JUL3117B", "SEP0417A"],
              "interrupt": ["MAR1517B", "JUL2717A", "AUG2517C", "AUG3017A",
                            "MAR0817B", "MAR1117A", "APR0217B", "SEP0917C"]}
all_loads = test_loads["normal"]+test_loads["interrupt"]

nlets = {"MAR0617A", "MAR0817B", "SEP0417A"}


def get_lr_root():
    """Get root directory for ACIS load review data.

    Try (in order):
    - /data/acis/LoadReviews
    - $SKA/data/acis/LoadReviews (for standalone installations)

    :returns: str, first path from above which exists.
    """
    data_acis_lr = Path('data', 'acis', 'LoadReviews')
    path = '/' / data_acis_lr
    if not path.exists():
        path = os.environ['SKA'] / data_acis_lr
        if not path.exists():
            raise FileNotFoundError('no available ACIS load review directory')
    return str(path)


class TestArgs(object):
    """
    A mock-up of a command-line parser object to be used with
    ACISThermalCheck testing.

    Parameters
    ----------
    name : string
        The "short name" of the temperature to be modeled.
    outdir : string
        The path to the output directory.
    model_path : string
        The path to the model code itself.
    run_start : string, optional
        The run start time in YYYY:DOY:HH:MM:SS.SSS format. If not
        specified, one will be created 3 days prior to the model run.
    load_week : string, optional
        The load week to be tested, in a format like "MAY2016". If not
        provided, it is assumed that a full set of initial states will
        be supplied.
    days : float, optional
        The number of days to run the model for. Default: 21.0
    T_init : float, optional
        The starting temperature for the run. If not set, it will be
        determined from telemetry.
    interrupt : boolean, optional
        Whether or not this is an interrupt load. Default: False
    state_builder : string, optional
        The mode used to create the list of commanded states. "sql" or
        "acis", default "acis".
    verbose : integer, optional
        The verbosity of the output. Default: 0
    model_spec : string, optional
        The path to the model specification file to use. Default is to
        use the model specification file stored in the model package.
    nlet_file : string, optional
        The path to an alternative NLET file to be used. Default: None,
        which is to use the default one. 
    """
    def __init__(self, name, outdir, model_path, run_start=None,
                 load_week=None, days=21.0, T_init=None, interrupt=False,
                 state_builder='acis', verbose=0, model_spec=None,
                 nlet_file=None):
        from datetime import datetime
        self.load_week = load_week
        if run_start is None:
            year = 2000 + int(load_week[5:7])
            month = months.index(load_week[:3])+1
            day = int(load_week[3:5])
            run_start = datetime(year, month, day).strftime("%Y:%j:%H:%M:%S")
        self.run_start = run_start
        self.outdir = outdir
        lr_root = get_lr_root()  # Directory containing ACIS load review data
        # load_week sets the bsdir
        if load_week is None:
            self.backstop_file = None
        else:
            load_year = "20%s" % load_week[-3:-1]
            load_letter = load_week[-1].lower()
            self.backstop_file = "%s/%s/%s/ofls%s" % (lr_root, load_year, load_week[:-1], load_letter)
        self.days = days
        if nlet_file is None:
            nlet_file = f'{lr_root}/NonLoadTrackedEvents.txt'
        self.nlet_file = nlet_file
        self.interrupt = interrupt
        self.state_builder = state_builder
        self.pred_only = False
        self.T_init = T_init
        self.traceback = True
        self.verbose = verbose
        if model_spec is None:
            model_spec = os.path.join(model_path, "%s_model_spec.json" % name)
        self.model_spec = model_spec
        self.version = None
        if name == "acisfp":
            self.fps_nopref = os.path.join(model_path, "FPS_NoPref.txt")


def exception_catcher(test, old, new, data_type, **kwargs):
    if new.dtype.kind == "S":
        new = new.astype("U")
    if old.dtype.kind == "S":
        old = old.astype("U")
    try:
        test(old, new, **kwargs)
    except AssertionError:
        raise AssertionError("%s are not the same!" % data_type)


class RegressionTester(object):
    def __init__(self, atc_class, model_path, model_spec, atc_args=None,
                 atc_kwargs=None, test_root=None, sub_dir=None):
        self.model_path = model_path
        if atc_args is None:
            atc_args = ()
        if atc_kwargs is None:
            atc_kwargs = {}
        self.atc_obj = atc_class(*atc_args, **atc_kwargs)
        self.msid = self.atc_obj.msid
        self.name = self.atc_obj.name
        self.valid_limits = self.atc_obj.validation_limits
        self.hist_limit = self.atc_obj.hist_limit
        self.curdir = os.getcwd()
        if test_root is None:
            rootdir = tempfile.mkdtemp()
        else:
            rootdir = test_root
        if sub_dir is not None:
            rootdir = os.path.join(rootdir, sub_dir)
        self.outdir = os.path.abspath(rootdir)
        self.test_model_spec = os.path.join(model_path, "tests", model_spec)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

    def run_model(self, load_week, run_start=None, state_builder='acis',
                  interrupt=False, override_limits=None):
        """
        Run a thermal model in test mode for a single load week.

        Parameters
        ----------
        load_week : string
            The load week to be tested, in a format like "MAY2016A".
        run_start : string, optional
            The run start time in YYYY:DOY:HH:MM:SS.SSS format. If not
            specified, one will be created 3 days prior to the model run.
        state_builder : string, optional
            The mode used to create the list of commanded states. "sql" or
            "acis", default "acis".
        interrupt : boolean, optional
            Whether or not this is an interrupt load. Default: False
        override_limits : dict, optional
            Override any margin by setting a new value to its name
            in this dictionary. SHOULD ONLY BE USED FOR TESTING.
        """
        out_dir = os.path.join(self.outdir, load_week)
        if load_week in nlets:
            nlet_file = os.path.join(os.path.dirname(__file__), 
                                     f'data/nlets/TEST_NLET_{load_week}.txt')
        else:
            nlet_file = None
        args = TestArgs(self.name, out_dir, self.model_path, run_start=run_start,
                        load_week=load_week, interrupt=interrupt, nlet_file=nlet_file,
                        state_builder=state_builder, model_spec=self.test_model_spec)
        self.atc_obj.run(args, override_limits=override_limits)

    def run_models(self, normal=True, interrupt=True, run_start=None,
                   state_builder='acis'):
        """
        Run the internally set list of models for regression testing.

        Parameters
        ----------
        normal : boolean, optional
            Run the "normal" loads. Default: True
        interrupt : boolean, optional
            Run the "interrupt" loads. Default: True
        run_start : string, optional
            The run start time in YYYY:DOY:HH:MM:SS.SSS format. If not
            specified, one will be created 3 days prior to the model run.
        state_builder : string, optional
            The mode used to create the list of commanded states. "sql" or
            "acis", default "acis".
        """
        if normal:
            for load in test_loads["normal"]:
                self.run_model(load, run_start=run_start,
                               state_builder=state_builder)
        if interrupt:
            for load in test_loads["interrupt"]:
                self.run_model(load, interrupt=True, run_start=run_start,
                               state_builder=state_builder)

    def _set_answer_dir(self, load_week):
        answer_dir = os.path.join(self.model_path, "tests/answers",
                                  load_week)
        if not os.path.exists(answer_dir):
            os.makedirs(answer_dir)
        return answer_dir

    def run_test(self, test_name, load_week, answer_store=False):
        """
        This method runs the answer test in one of two modes:
        either comparing the answers from this test to the "gold
        standard" answers or to simply run the model to generate answers.

        Parameters
        ----------
        test_name : string
            The name of the test to run. "prediction" or "validation".
        load_week : string
            The load week to be tested, in a format like "MAY2016A".
        answer_store : boolean, optional
            If True, store the generated data as the new answers.
            If False, only test. Default: False
        """
        out_dir = os.path.join(self.outdir, load_week)
        if test_name == "prediction":
            filenames = ["temperatures.dat", "states.dat"]
            if self.name == "acisfp":
                filenames.append("earth_solid_angles.dat")
        elif test_name == "validation":
            filenames = ["validation_data.pkl"]
        else:
            raise RuntimeError("Invalid test specification! "
                               "Test name = %s." % test_name)
        if not answer_store:
            compare_test = getattr(self, "compare_"+test_name)
            compare_test(load_week, out_dir, filenames)
        else:
            answer_dir = self._set_answer_dir(load_week)
            self.copy_new_files(out_dir, answer_dir, filenames)

    def compare_validation(self, load_week, out_dir, filenames):
        """
        This method compares the "gold standard" validation data 
        with the current test run's data.

        Parameters
        ----------
        load_week : string
            The load week to be tested, in a format like "MAY2016A".
        out_dir : string
            The path to the output directory.
        filenames : list of strings
            The list of files which will be used in the comparison.
            Currently only "validation_data.pkl".
        """
        # First load the answers from the pickle files, both gold standard
        # and current
        new_answer_file = os.path.join(out_dir, filenames[0])
        new_results = pickle.load(open(new_answer_file, "rb"))
        old_answer_file = os.path.join(self.model_path, "tests/answers", load_week,
                                       filenames[0])
        old_results = pickle.load(open(old_answer_file, "rb"))
        # Compare predictions
        new_pred = new_results["pred"]
        old_pred = old_results["pred"]
        pred_keys = set(new_pred.keys()) | set(old_pred.keys())
        for k in pred_keys:
            if k not in new_pred:
                print("WARNING in pred: '%s' in old answer but not new. Answers should be updated." % k)
                continue
            if k not in old_pred:
                print("WARNING in pred: '%s' in new answer but not old. Answers should be updated." % k)
                continue
            exception_catcher(assert_allclose, new_pred[k], old_pred[k],
                              "Validation model arrays for %s" % k, rtol=1.0e-5)
        # Compare telemetry
        new_tlm = new_results['tlm']
        old_tlm = old_results['tlm']
        tlm_keys = set(new_tlm.dtype.names) | set(old_tlm.dtype.names)
        for k in tlm_keys:
            if k not in new_tlm.dtype.names:
                print("WARNING in tlm: '%s' in old answer but not new. Answers should be updated." % k)
                continue
            if k not in old_tlm.dtype.names:
                print("WARNING in tlm: '%s' in new answer but not old. Answers should be updated." % k)
                continue
            exception_catcher(assert_array_equal, new_tlm[k], old_tlm[k],
                              "Validation telemetry arrays for %s" % k)

    def compare_prediction(self, load_week, out_dir, filenames):
        """
        This method compares the "gold standard" prediction data with 
        the current test run's data for the .dat files produced in the 
        thermal model run.

        Parameters
        ----------
        load_week : string
            The load week to be tested, in a format like "MAY2016A".
        out_dir : string
            The path to the output directory.
        filenames : list of strings
            The list of files which will be used in the comparison.
        """
        from astropy.io import ascii
        for fn in filenames:
            new_fn = os.path.join(out_dir, fn)
            old_fn = os.path.join(self.model_path, "tests/answers", load_week, fn)
            new_data = ascii.read(new_fn).as_array()
            old_data = ascii.read(old_fn).as_array()
            # Compare test run data to gold standard. Since we're loading from
            # ASCII text files here, floating-point comparisons will be different
            # at machine precision, others will be exact.
            for k, dt in new_data.dtype.descr:
                if 'f' in dt:
                    exception_catcher(assert_allclose, new_data[k], old_data[k],
                                      "Prediction arrays for %s" % k, rtol=1.0e-5)
                else:
                    exception_catcher(assert_array_equal, new_data[k], old_data[k],
                                      "Prediction arrays for %s" % k)
 
    def copy_new_files(self, out_dir, answer_dir, filenames):
        """
        This method copies the files generated in this test
        run to a directory specified by the user, typically for
        inspection and for possible updating of the "gold standard"
        answers.

        Parameters
        ----------
        out_dir : string
            The path to the output directory.
        answer_dir : string
            The path to the directory to which to copy the files.
        filenames : list of strings
            The filenames to be copied.
        """
        for filename in filenames:
            fromfile = os.path.join(out_dir, filename)
            tofile = os.path.join(answer_dir, filename)
            shutil.copyfile(fromfile, tofile)

    def check_violation_reporting(self, load_week, viol_json, 
                                  answer_store=False):
        """
        This method runs loads which report violations of
        limits and ensures that they report the violation,
        as well as the correct start and stop times.

        Parameters
        ----------
        load_week : string
            The load to check. 
        model_spec : string
            The path to the model specification file to
            use. For this test, to ensure the violation is
            reported in the same way, we must use the same
            model specification file that was used at the
            time of the run.
        viol_json : string
            Path to the JSON file containing the answers
            for the violation data.
        answer_store : boolean, optional
            If True, store the generated data as the new answers.
            If False, only test. Default: False
        """
        import json
        with open(viol_json, "r") as f:
            viol_data = json.load(f)
        if answer_store:
            viol_data["datestarts"] = []
            viol_data["datestops"] = []
            viol_data["temps"] = []
            if self.msid == "fptemp":
                viol_data["obsids"] = []
        load_year = "20%s" % load_week[-3:-1]
        self.run_model(load_week, run_start=viol_data['run_start'], 
                       override_limits=viol_data['limits'])
        out_dir = os.path.join(self.outdir, load_week)
        index_rst = os.path.join(out_dir, "index.rst")
        with open(index_rst, 'r') as myfile:
            i = 0
            for line in myfile.readlines():
                if line.startswith("Model status"):
                    assert "NOT OK" in line
                if line.startswith(load_year):
                    if answer_store:
                        words = line.strip().split()
                        viol_data["datestarts"].append(words[0])
                        viol_data["datestops"].append(words[1])
                        viol_data["temps"].append(words[2])
                        if self.msid == "fptemp":
                            viol_data["obsids"].append(words[3])
                    else:
                        try:
                            assert viol_data["datestarts"][i] in line
                            assert viol_data["datestops"][i] in line
                            assert viol_data["temps"][i] in line
                            if self.msid == "fptemp":
                                assert viol_data["obsids"][i] in line
                        except AssertionError:
                            raise AssertionError("Comparison failed. Check file at "
                                                 "%s." % index_rst)
                    i += 1
        if answer_store:
            with open(viol_json, "w") as f:
                json.dump(viol_data, f, indent=4)

