
import numpy as np
class Patient:
    def __init__(self):
        self.CandidateList = []
        self.IMG = np.empty(1)
        self.Origin = np.empty(1)
        self.Spacing = np.empty(1)
        
    def setCandidateList(self, XYZ, Label):
        infodict = {}
        infodict['XYZ'] = XYZ
        infodict['Label'] = Label[:-2]
        self.CandidateList.append(infodict)
    def setIMG(self, img):
        self.IMG = img.copy()
    def setOrigin(self, origin):
        self.Origin = origin.copy()
    def setSpacing(self, spacing):
        self.Spacing = spacing.copy()