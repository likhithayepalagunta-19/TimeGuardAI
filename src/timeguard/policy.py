from dataclasses import dataclass


@dataclass
class MaintenancePolicy:
    horizon: float = 45
    state: str = 'OBSERVE'
    low_count: int = 0
    safe_count: int = 0

    def update(self, lower, valid=True):
        if not valid:
            self.low_count = self.safe_count = 0
            self.state = 'REVIEW'
            return self.state
        if lower <= self.horizon:
            self.low_count += 1
            self.safe_count = 0
            if self.low_count >= 3:
                self.state = 'PLAN_MAINTENANCE'
        elif lower > self.horizon+10:
            self.low_count = 0
            self.safe_count += 1
            if self.safe_count >= 5:
                self.state = 'OBSERVE'
        else:
            self.low_count = self.safe_count = 0
        return self.state
