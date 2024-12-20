class Simulator():
    def __init__(self,inputs):
        self.temp = inputs[0]
        self.magstr = inputs[1]
        self.tritflow = inputs[2]
        self.powerin = inputs[3]
        self.powerout = inputs[4]
        self.h4flow = inputs[5]
        self.neutronflux = inputs[6]
        self.coolantflow = inputs[7]
        self.fuelin = inputs[8]
        self.rad = inputs[9]

        self.base_efficiency = 0.85  
        self.heat_loss_factor = 0.1  
        self.max_temp = 20  
        self.max_mag_conf = 10
    def sim(self):
        fusion_energy = self.temp * self.magstr * self.tritflow 
        net_energy = fusion_energy - self.rad
        
        cooling_loss = self.coolantflow * (1 - self.base_efficiency)

            # Final power output and losses
        output_energy = self.base_efficiency * net_energy - cooling_loss
        total_loss = self.powerin - output_energy + cooling_loss + self.heat_loss_factor * fusion_energy

            # Stability and efficiency calculations
        stability = 1 - abs(self.powerin - output_energy) / (self.powerin + 1e-6)
        efficiency = output_energy / self.powerin if self.powerin > 0 else 0

        return [output_energy, total_loss, stability, efficiency]
        



