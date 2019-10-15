from collections import defaultdict
from metadata import Metadata
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor

"""
self.beads should be what?
Dictionary for each hybe
An item does not exist if finding beads has never been attempted, but if 
finding beads has been attempted then it should contain whether it passed for failed

('Pass/Fail', if Pass: number beads found else: Failure reason, dict(extra info))

self.tforms should be what?
('Pass/Fail', if Pass: tform else Fail Reason, dict(extra info))
"""

class MerfishMetadata(object):
    """
    Class that defines experiment specific aspects of MERFISH experiments.
    
    Also containers helper functions to extracting information from the 
    related metadata.
    """
    def __init__(self, config, config_type = 'python_module', hybe_names = 'default9'):
        """
        Long term this should accept a config module or a json ala starfish.
        It should import the metadata and then check for valid parameters by 
        throwing an exception if a minimum set of valid class attributes 
        aren't defined at the end of the init.
        
        For example:
        self.bitmap
        self.normalized_codevectors
        self.codevectors
        self.gene_names
        self.transcript_ids
        self.image_dimensions
        self.chromatic_map
        self.pixel_size
        etc... All the fields that our standard pipeline assumes are present
        Non-standard experiments (for example tissue/cornea that might require
        extra information can subclass this although that probably isn't something 
        that should happen much for MerfishMetadata and will be more relevant for other
        classes such as Registration or Classification classes.
        """
        if hybe_names == 'default9':
            self.hybe_names = np.array(["hybe{0}".format(i+1) for i in range(9)])
        elif isinstance(hybe_names, list):
            self.hybe_names = hybe_names
        else:
            raise ValueError('Only implemented defaults and list are valid input for hybe_names argument')
        if isinstance(config, str):
            self.config_module = self._import_config(config, config_format=config_type)
        else:
            raise ValueError('Only strings naming a python module or json file is valid value for config argument.')
            
    def get_hybe_codestack_idx(self, hybe):
        return np.array([i for i, b in enumerate(bitmap) if b[1]==hybe])
    
    def get_cword_one_bits(self, i):
        return np.where(self.normalized_codevectors>0)[0]
    
    def _import_config(self, config_name, config_format='python_module'):
        """
        Import module configuration.
        
        Notes:
        Must define the following module properties should implement checking in __init__:
        bitmap, normalized_codevectors, codevectors, gene_names, transcript_ids
        """
        if config_format=='python_module':
            config_module = __import__(config_name)
            self.bitmap = config_module.bitmap
            self.normalized_codevectors = config_module.norm_gene_codeword_vectors
            self.codevectors = np.stack(config_module.cwords, axis=0)
        elif config_format == 'starfish_json':
            raise ValueError('Not Implemented')
            
class RegistrationContainer(object):
    """
    Container for MERFISH image registration for single FOV.
    """
    def __init__(self, position_name, experiment, stageXY, beads={}, tforms={},
                 residuals = {}, matched_beads = {}, residual_threshold=0.5, number_beads_threshold = 20):
        """
        A FOV has:
        stageXY
        beads
        tforms
        residuals
        matched_beads
        
        Registration has some parameters:
        residual_threshold
        number_beads_threshold
        """
        self.position_name = position_name
        # Dict/BeadContainer
        self.beads = beads
        # Dict/TransformContainer
        self.tforms = tforms
        self.experiment = experiment
#         self.hybe_names = self.experiment.hybe_names
        self.stageXY = stageXY
        self.bead_residuals = residuals
        self.matched_bead_counts = matched_beads
        self._check_status()

    def _check_status(self):
        """
        Iterate over relevant fields and flag if they don't meet minimum specifications.
        
        Implemented:
        non-nan transformations (nan if not enough beads found) this will probably deprecate as we move away from old format
        matched_beads > t
        residual < t
        """
        self.flag = {h: 'good' for h in tforms.keys()}
        for h, v in tforms.items():
            if any(np.isnan(v)):
                self.flag[h] = 'bad'
        for h, v in self.bead_residuals.items():
            if v>residual_threshold:
                self.flag[h] = 'bad'
        for h, v in self.matched_bead_counts.items():
            if v<number_beads_threshold:
                self.flag[h]='bad'

    def __getitem__(self, i):
        """
        Easier access. Should make these properties so they're accessible with object.field notation.
        """
        if i == 'beads':
            return self.beads
        elif i == 'tforms':
            return self.tforms
        elif i == 'xy':
            return self.stageXY
        elif i == 'name':
            return self.position_name
        else:
            raise LookupError('Only beads, tforms, and xy items are available by index')


class RegistrationModel2D(object):
    """
    Fits model of Registration for multiple FOVs.
    
    The model can predict an expected translation based on stage XY coordinates.
    """
    def __init__(self, stageXY, translations):
        self.xy = stageXY
        self.translations = translations
        self._nans = np.isnan(self.translations.mean(axis=1))
    def fill_nans(self, method='knn'):
        # If there are no nans just return the bead fitted values
        if sum(self._nans)==0:
            self.predicted_translations = self.translations
            return self.translations
        # Switch for different value filling methods
        if method == 'knn':
            knnr = KNeighborsRegressor(weights='distance', n_neighbors=6)
            knnr.fit(self.xy[~self._nans, :], self.translations[~self._nans, :])
            predicted = self.translations.copy()
            predicted[self._nans, :] = knnr.predict(self.xy[self._nans, :])
            self.predicted_translations = predicted
            return predicted
        else:
            raise ValueError('Invalid method type supplied.')
    def _weights(self, dists):
        return 'Not Implemented'

class MultiPositionRegistration(object):
    def __init__(self, fov_registration_objects):
        for fov in fov_registration_objects:
            if not isinstance(fov, RegistrationContainer):
                raise TypeError('Input FOV is not a RegistrationCountainer')
        self._fovs = fov_registration_objects
        
    def iterate_over_hybes_and_fit(self, destination_hybe = 'hybe1', fov_subset = None, verbose=True):
        hybes = ['hybe1', 'hybe2', 'hybe3', 'hybe3', 'hybe4', 'hybe5', 'hybe6', 'hybe7', 'hybe8', 'hybe9']
        hybes.remove(destination_hybe)
        if fov_subset is None:
            fovs = self._fovs
        else:
            fovs = fov_subset
        has_nans = defaultdict(dict)
        predicted = defaultdict(dict)
        models = defaultdict(dict)
        for idx, f in enumerate(self._fovs):
            predicted[f['name']][destination_hybe] = np.zeros((1, 3))
        for hybe in hybes:
            for idx, f in enumerate(self._fovs):
                model = self._fit_2d_registration(destination_hybe, hybe, fov_subset=fov_subset, verbose=verbose)
                if f.flag[hybe]=='bad':
                    has_nans[f['name']][hybe] = model
                predicted[f['name']][hybe] = model.predicted_translations[idx]
                models[f['name']][hybe] = model
        return predicted, has_nans, models
        
    def _fit_2d_registration(self, hybe_destination, hybe_source, fov_subset = None, verbose=True):
        """
        Fit the global trend in offsets accross multiple FOVs.
        
        Parameters
        ----------
        hybe_source : name of hybe to register from
        hybe_destination : name of hybe to register to
        fov_subset : list of FOVs to use default(all)
        
        Returns
        -------
        model : fitted 2D model
        """
        
        stage_coordinates = []
        estimated_translations = []
        
        if fov_subset is None:
            fovs = self._fovs
        else:
            fovs = fov_subset
        # Iterate over fovs and pull out info for specific hybes
        for f in fovs:
            stage_coordinates.append(f['xy'])
            tforms = f['tforms']
            if (hybe_source not in tforms) or (hybe_destination not in tforms):
                if verbose:
                    print(tforms)
                    print("Warning tform doesn't exist for position: {0}".format(f['name']))
                t = np.nan
            else:
                t = tforms[hybe_destination] - tforms[hybe_source]
            estimated_translations.append(t)
        stageXY, translations = np.stack(stage_coordinates, axis=0), np.stack(estimated_translations, axis=0)
        model = RegistrationModel2D(stageXY, translations)
        model.fill_nans()
        for idx, f in enumerate(fovs):
            f['tforms'][hybe_source] = model.predicted_translations[idx]
        return model
        

        
def from_goodBad_format_to_container_objects(tforms):
    all_tforms = {}
    residuals = {}
    matched_bead_counts = {}
    for p, t in tforms['good'].items():
        all_tforms[p] = {h: t[0] for h, t in t.items()}
        residuals[p] = {h: t[1] for h, t in t.items()}
        matched_bead_counts[p] = {h: t[2] for h, t in t.items()}
    for p, t in tforms['bad'].items():
        new_t = {}
        new_r = {}
        new_bc = {}
        for h, t in t.items():
            if isinstance(t[0], str):
                new_t[h] = (np.nan, np.nan, np.nan)
                new_r[h] = np.nan
                new_bc[h] = np.nan
            else:
                new_t[h] = t[0]
                new_r[h] = t[1]
                new_bc[h] = t[2]
        all_tforms[p] = new_t
        residuals[p] = new_r
        matched_bead_counts[p] = new_bc
    return all_tforms, residuals, matched_bead_counts


    
