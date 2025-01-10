class Simulator():
    history = [15,30,0.042]
    def __init__(self, tempst, magstr, tritflow, powerin, coolantflow):
        # Initialize the simulator with the input parameters
        self.tempst = tempst
        self.magstr = magstr
        self.tritflow = tritflow
        self.powerin = powerin
        self.coolantflow = coolantflow
        self.history = [15,30,0.042]


    def simulate(self):
        htemp = self.history[0]
        hrad_loss = self.history[1]
        hefficiency = self.history[2]

        # Example of how outputs might be derived (this is a placeholder for actual fusion reactor physics)
        powerout = hefficiency * self.powerin * self.tritflow / self.tempst# Placeholder for actual calculations
        rad_loss = (hrad_loss + (self.tempst * self.magstr * 0.05))/2  # Placeholder for radiation loss calculation
        
        # Efficiency is calculated as power output divided by input power
        efficiency = powerout / self.powerin if self.powerin != 0 else 0

        temp = (((htemp + self.tempst)/2) + htemp)/2

        self.history = [temp, rad_loss, efficiency]

        return temp, rad_loss, efficiency

    def update_inputs(self, temp=None, magstr=None, tritflow=None, powerin=None, coolantflow=None):
        # Update the inputs if new values are provided
        if temp is not None:
            self.temp = temp
        if magstr is not None:
            self.magstr = magstr
        if tritflow is not None:
            self.tritflow = tritflow
        if powerin is not None:
            self.powerin = powerin
        if coolantflow is not None:
            self.coolantflow = coolantflow

    def get_inputs(self):
        # Return the current input values
        return [self.tempst,
         self.magstr,
         self.tritflow,
         self.powerin,
        self.coolantflow
        ]