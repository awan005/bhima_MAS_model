__author__ = 'ankun wang'

class IntegratedModelFrame:
    """
    Frame of the dynamic hydrological model

    lastTimeStep:  Last time step to run
    firstTimestep: Starting time step of the model
    """

    def __init__(self, model, lastTimeStep=0, firstTimestep=0, parameter=0,Name='DateYears'):
        """
        sets first and last time step into the model

        :param lastTimeStep: last timestep
        :param firstTimeStep: first timestep
        :return: -
        """
        self._model = model
        self._model.lastStep = lastTimeStep
        self._model.firstStep = firstTimestep
        self._model.para = parameter
        self._model.name = Name

    def step(self):

        self._model.currentStep = self.currentStep
        if (self._model.currentStep % 12) == 0:
            self._model.dynamicstage()
        else:
            pass
        self.currentStep += 1

    def initialize_run(self):
        # inside cwatm_dynamic it will interate
        self.currentStep = self._model.firstStep
        self.para = self._model.para
        self.name = self._model.para

    def run(self):
        """  Run the dynamic part of the model

        :return: -
        """
        self.initialize_run()
        while self.currentStep <= self._model.lastStep:
            self.step()







