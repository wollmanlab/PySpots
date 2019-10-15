from collections import defaultdict
from metadata import Metadata

"""
self.beads should be what?
Dictionary for each hybe
An item does not exist if finding beads has never been attempted, but if 
finding beads has been attempted then it should contain whether it passed for failed

('Pass/Fail', if Pass: number beads found else: Failure reason, dict(extra info))

self.tforms should be what?
('Pass/Fail', if Pass: tform else Fail Reason, dict(extra info))
"""

# class MerfishExperimentStatus(object):
#     def __init__(self, metadata_path, zrange=None):
#         self.metadata_path = metadata_path
#         self.metadata = Metadata(self.metadata_path)
#         self.zrange=zrange
        
#     def find_completed_positions(self, refresh_metadata = False):
#         """
#         Return all completed/synced hybe/position pairs.
        
#         """
#         return 'Not Implemented'
#         if refresh_metadata:
#             self.metadata = Metadata(self.metadata_path)
#         for (hybe, posname), subdf in self.metadata.groupby(''):
#             pass
       

class MerfishMetadata(object):
    def __init__(self, bitmap, hybe_names = 'default9'):
        if hybe_names == 'default9':
            self.hybe_names = np.array(["hybe{0}".format(i+1) for i in range(9)])
        else:
            self.hybe_names = hybe_names
        if isinstance(bitmap, str):
            self.config_module = self._import_config(bitmap)
        else:
            self.bitmap = bitmap
            
    def get_hybe_codestack_idx(self, hybe):
        return np.array([i for i, b in enumerate(bitmap) if b[1]==hybe])
    def get_cword_one_bits(self, i):
        return np.where(self.normalized_codevectors>0)[0]
    
    def _import_config(self, config_name, config_format='python_module'):
        """
        Import module configuration.
        
        Notes:
        Must define the following module properties:
        bitmap, normalized_codevectors, codevectors, gene_names, transcript_ids
        """
        if config_format=='python_module':
            config_module = __import__(config_name)
            self.bitmap = config_module.bitmap
            self.normalized_codevectors = config_module.norm_gene_codeword_vectors
            self.codevectors = np.stack(config_module.cwords, axis=0)
        elif config_format == 'starfish_json':
            raise ValueError('Not Implemented')

class RegistrationModel2D(object):
    """
    Fits model of Registration for multiple FOVs.
    
    The model can predict an expected translation based on stage XY coordinates.
    """
    def __init__(self):
        pass

class MultiPositionRegistration(object):
    def __init__(self, fov_registration_objects):
        for fov in fov_registration_objects:
            if not isinstance(fov, RegistrationContainer):
                raise TypeError('Input FOV is not a RegistrationCountainer')
        self._fovs = fov_registration_objects
        
    def _fit_2d_registration(self, hybe_source, hybe_destination, fov_subset = None):
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
        model = RegistrationModel2D()
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
                    print("Warning tform doesn't exist for position: {0}".format(f['name']))
                t = np.nan
            else:
                t = tforms[hybe_destination] - tforms[hybe_source]
            estimated_translations.append(t)
        return stage_coordinates, esimated_translations
        model.fit(stage_coordinates, 
                        
        
        return 'Not Implemented'

                                                            
class RegistrationContainer(object):
    def __init__(self, position_name, experiment, stageXY):
        self.position_name = position_name
        # Dict/BeadContainer
        self.beads = {}
        # Dict/TransformContainer
        self.tforms = {}
        self.experiment = experiment
        self.hybe_names = self.experiment.hybe_names
        self.stageXY = stageXY
        
    def __getitem__(self, i):
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
        
#     @property
#     def get_status(self):
#         """
#         Return the status of the registration for a position.
        
#         Returns
#         -------
        
        
#         Notes:
#         Important status aspects for registration status:
#         1. Hybes completed
#             a. finding beads
#             b. calculating tform
#         2. If there is a failure
#             a. Which hybes failed
#             b. Why did they fail
#                 i. Not enough beads
#                 ii. Residual too high
#                 iii. error
#         """
#         return 'Not implemented'
#         beads_status = defaultdict(dict)
#         tforms_status = defaultdict(dict)
#         for hybe in self.hybe_names:
#             if hybe in self.beads:
#                 if 
#                 beads_status['completed'][hybe] = len(self.beads['hybe'])
#             else:
#                 beads_status['incomplete'][hybe] = 'not implemented'
        
        
    
