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

class MerfishExperimentStatus(object):
    def __init__(self, metadata_path, zrange=None):
        self.metadata_path = metadata_path
        self.metadata = Metadata(self.metadata_path)
        self.zrange=zrange
        
    def find_completed_positions(self, refresh_metadata = False):
        """
        Return all completed/synced hybe/position pairs.
        
        """
        if refresh_metadata:
            self.metadata = Metadata(self.metadata_path)
        for (hybe, posname), subdf in self.metadata.groupby('
class MerfishExperiment(object):
    def __init__(self, image):
        pass

class RegistrationContainer(object):
    def __init__(self, position_name, experiment):
        self.position_name = position_name
        self.beads = None
        self.tform = None
        self.experiment = experiment
        self.hybe_names = self.experiment.hybe_names
        
    @property
    def get_status(self):
        """
        Return the status of the registration for a position.
        
        Returns
        -------
        
        
        Notes:
        Important status aspects for registration status:
        1. Hybes completed
            a. finding beads
            b. calculating tform
        2. If there is a failure
            a. Which hybes failed
            b. Why did they fail
                i. Not enough beads
                ii. Residual too high
                iii. error
        """
        beads_status = defaultdict(dict)
        tforms_status = defaultdict(dict)
        for hybe in self.hybe_names:
            if hybe in self.beads:
                if 
                beads_status['completed'][hybe] = len(self.beads['hybe'])
            else:
                beads_status['incomplete'][hybe] = 
        
        return 'Not implemented'
    
