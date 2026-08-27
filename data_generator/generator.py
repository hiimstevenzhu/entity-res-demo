import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any

# SciPy ≥ 1.6 provides scipy.stats.loguniform; otherwise we fall back
# to a manual implementation (exponential of a uniform).
try:
    from scipy.stats import loguniform   # type: ignore
except Exception:  # pragma: no cover
    loguniform = None  # we’ll handle it below


class DataTypeGenerator(ABC):
    """Abstract Base Class acting as the Factory interface for Data Types."""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng

    @abstractmethod
    def generate_base(self, size: int, config: Dict[str, Any]) -> np.ndarray:
        """
        Generate latent ground-truth values (no noise, no missingness) for `size` entities.
        :param size: Number of entities.
        :param config: Feature configuration.
        :return: NumPy array of shape (size,) with latent values.
        """
        pass

    @abstractmethod
    def apply_noise(self, base_values: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Apply noise and missingness to latent base values.
        :param base_values: NumPy array of latent values (shape (n_rows,)).
        :param config: Feature configuration.
        :return: NumPy array of noisy/missing values.
        """
        pass

    def _apply_missingness(self, data: np.ndarray, missing_rate: float, missing_value: Any = np.nan) -> np.ndarray:
        """Vectorised helper to inject missing values (NaN/None) across the array."""
        if missing_rate <= 0.0:
            return data

        mask = self.rng.random(len(data)) < missing_rate

        if isinstance(data, pd.Series):
            data = data.copy()
            data[mask] = missing_value
            return data
        else:
            out = data.copy()
            out[mask] = missing_value
            return out


class CategoricalGenerator(DataTypeGenerator):
    """Generates categorical data."""

    def generate_base(self, size: int, config: Dict[str, Any]) -> np.ndarray:
        num_classes = config["num_classes"]
        classes = np.array([f"CAT_{i}" for i in range(num_classes)], dtype=object)
        return self.rng.choice(classes, size=size)

    def apply_noise(self, base_values: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        # Categorical: only missingness, no noise
        missing_rate = config.get("missing_rate", 0.0)
        return self._apply_missingness(base_values, missing_rate, missing_value=None)


class NumericalGenerator(DataTypeGenerator):
    """Generates numerical fields with optional Gaussian noise."""

    # def generate_base(self, size: int, config: Dict[str, Any]) -> np.ndarray:
    #     # Latent Gaussian with mean=255 and std=141.45 to match uniform [10,500] variance
    #     #return self.rng.normal(loc=255.0, scale=141.45, size=size)
    #     # Skewed distribution: log-normal with mean = 200 and sigma = 5, for variance = 25,
    #     target_mode = 100
    #     sigma = 0.8
    #     mu = np.log(target_mode) + sigma**2
    #     return self.rng.lognormal(mean=mu, sigma=sigma, size=size)

    def generate_base(self, size: int, config: Dict[str, Any]) -> np.ndarray:
        """
        Log‑uniform (reciprocal) distribution:
            - strict non‑negative values
            - mode (highest density) at the lower bound `low`
            - hard upper bound `high`
            - equal probability per decade → “fat” right‑tail

        Config keys (all optional, with sensible defaults):
            low    (float): left boundary – where the mode sits (default 150.0)
            high   (float): right boundary – hard cut‑off (default 3000.0)
            seed   (int):   if you want to override the Generator’s seed for this call
        """
        rng = self.rng  # the Generator supplied by the base class

        low  = float(config.get("low", 150.0))
        high = float(config.get("high", 3000.0))

        if low <= 0.0:
            raise ValueError("Log‑uniform requires low > 0.")
        if high <= low:
            raise ValueError("high must be greater than low.")

        # ------------------------------------------------------------------
        # Use SciPy’s loguniform when available; otherwise draw from
        #   X = exp( U )  where U ~ Uniform(ln(low), ln(high))
        # ------------------------------------------------------------------
        print("lopguniform is not None:", loguniform is not None)
        if loguniform is not None:
            dist = loguniform(low, high)   # SciPy’s built‑in
            return dist.rvs(size=size, random_state=rng)
        else:
            # Manual fallback – works with any NumPy version
            u = rng.uniform(low=np.log(low), high=np.log(high), size=size)
            return np.exp(u)


    def apply_noise(self, base_values: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        output = base_values.copy().astype(float)
        noise_type = config.get("noise_type", "none")
        if noise_type == "gaussian":
            sigma = config.get("noise_sigma", 1.0)
            noise = self.rng.normal(loc=0.0, scale=sigma, size=len(output))
            output += noise
        missing_rate = config.get("missing_rate", 0.0)
        return self._apply_missingness(np.round(output,2), missing_rate, missing_value=np.nan) # we do round to bring it to 2 d.p


class DateTimeGenerator(DataTypeGenerator):
    """Generates timestamps with optional Gaussian time-shifted noise."""

    def generate_base(self, size: int, config: Dict[str, Any]) -> np.ndarray:
        start_ts = pd.to_datetime(config["start_date"]).value
        end_ts = pd.to_datetime(config["end_date"]).value
        return self.rng.integers(start_ts, end_ts, size=size, dtype=np.int64)

    def apply_noise(self, base_values: np.ndarray, config: Dict[str, Any]) -> pd.Series:
        # Convert latent ints to datetime
        ts = pd.Series(pd.to_datetime(base_values))
        noise_type = config.get("noise_type", "none")
        if noise_type == "gaussian":
            sigma_hours = config.get("noise_sigma_hours", 2.0)
            sigma_seconds = sigma_hours * 3600.0
            noise_seconds = self.rng.normal(loc=0.0, scale=sigma_seconds, size=len(ts))
            ts += pd.to_timedelta(noise_seconds, unit='s')
        missing_rate = config.get("missing_rate", 0.0)
        return self._apply_missingness(ts, missing_rate, missing_value=pd.NaT)


class FirstNameGenerator(DataTypeGenerator):
    """Generates first names with noise including typos, nicknames, and truncation/elongation."""

    # Common US first names (mix of traditional and modern)
    FIRST_NAMES = [
        'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
        'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
        'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Nancy', 'Daniel', 'Lisa',
        'Matthew', 'Betty', 'Anthony', 'Helen', 'Donald', 'Sandra', 'Mark', 'Donna',
        'Steven', 'Carol', 'Paul', 'Ruth', 'Andrew', 'Sharon', 'Joshua', 'Michelle',
        'Kenneth', 'Laura', 'Kevin', 'Sarah', 'Brian', 'Kimberly', 'George', 'Deborah',
        'Edward', 'Dorothy', 'Ronald', 'Lisa', 'Timothy', 'Nancy', 'Jason', 'Samantha',
        'Jeffrey', 'Maria', 'Ryan', 'Ashley', 'Jacob', 'Emily', 'Gary', 'Melissa',
        'Nicholas', 'Andrea', 'Eric', 'Megan', 'Jonathan', 'Kelly', 'Stephen', 'Jessica',
        'Larry', 'Anna', 'Justin', 'Michelle', 'Scott', 'Kimberly', 'Brandon', 'Donna',
        'Benjamin', 'Carol', 'Samuel', 'Michelle', 'Gregory', 'Laura', 'Alexander', 'Sarah',
        'Patrick', 'Laura', 'Frank', 'Michelle', 'Raymond', 'Laura', 'Jack', 'Laura'
    ]

    # Common nicknames for first names
    NICKNAMES = {
        'James': ['Jim', 'Jimmy', 'Jamie'],
        'John': ['Jack', 'Johnny', 'Jon'],
        'Robert': ['Bob', 'Bobby', 'Rob'],
        'Michael': ['Mike', 'Mikey', 'Mickey'],
        'William': ['Will', 'Bill', 'Billy', 'Willy'],
        'David': ['Dave', 'Davey'],
        'Richard': ['Rich', 'Rick', 'Ricky', 'Dick'],
        'Joseph': ['Joe', 'Joey'],
        'Thomas': ['Tom', 'Tommy'],
        'Charles': ['Chuck', 'Charlie'],
        'Christopher': ['Chris', 'Topher'],
        'Daniel': ['Dan', 'Danny'],
        'Matthew': ['Matt'],
        'Anthony': ['Tony'],
        'Donald': ['Don', 'Donny'],
        'Mark': ['Marc'],
        'Steven': ['Steve', 'Steve'],
        'Paul': ['Paulo'],
        'Andrew': ['Andy', 'Drew'],
        'Joshua': ['Josh'],
        'Kenneth': ['Ken', 'Kenny'],
        'Kevin': ['Kev'],
        'Brian': ['Bri'],
        'George': ['Geo'],
        'Edward': ['Ed', 'Eddie', 'Ted'],
        'Ronald': ['Ron', 'Ronnie'],
        'Timothy': ['Tim'],
        'Jason': ['Jay'],
        'Jeffrey': ['Jeff'],
        'Ryan': ['Ry'],
        'Jacob': ['Jake'],
        'Gary': ['Gaz'],
        'Nicholas': ['Nick', 'Nicky'],
        'Eric': ['Rick'],
        'Jonathan': ['Jon'],
        'Stephen': ['Steve', 'Stefan'],
        'Larry': ['Larz'],
        'Justin': ['Just'],
        'Scott': ['Scotty'],
        'Brandon': ['Brand'],
        'Benjamin': ['Ben', 'Benny'],
        'Samuel': ['Sam', 'Sammy'],
        'Gregory': ['Greg', 'Gregg'],
        'Alexander': ['Alex', 'Xander'],
        'Patrick': ['Pat', 'Patty'],
        'Frank': ['Franky'],
        'Raymond': ['Ray'],
        'Jack': ['Jackie'],
        'Mary': ['Molly', 'Polly'],
        'Patricia': ['Pat', 'Patsy', 'Trisha'],
        'Jennifer': ['Jen', 'Jenny', 'Jenna'],
        'Linda': ['Lin', 'Lynda'],
        'Elizabeth': ['Liz', 'Beth', 'Betty', 'Eliza'],
        'Barbara': ['Barb', 'Bobbi'],
        'Susan': ['Susie', 'Suzy'],
        'Jessica': ['Jess', 'Jessie'],
        'Sarah': ['Sara', 'Sassie'],
        'Karen': ['Kar', 'Kari'],
        'Nancy': ['Nan'],
        'Lisa': ['Lee'],
        'Karen': ['Kar'],
        'Betty': ['Bet'],
        'Helen': ['Helly', 'Lena'],
        'Sandra': ['Sandie', 'Sandy'],
        'Donna': ['Donnie'],
        'Carol': ['Caro'],
        'Ruth': ['Ruthie'],
        'Sharon': ['Shari'],
        'Michelle': ['Mickey', 'Shelly'],
        'Laura': ['Laurie'],
        'Kimberly': ['Kim', 'Kimi'],
        'Deborah': ['Deb', 'Debbie'],
        'Dorothy': ['Dot', 'Dotty'],
        'Lisa': ['Lee'],
        'Nancy': ['Nan'],
        'Samantha': ['Sam', 'Sammy'],
        'Maria': ['Mari', 'Mar'],
        'Ashley': ['Ash'],
        'Emily': ['Em', 'Emmy'],
        'Melissa': ['Missy'],
        'Andrea': ['Andie'],
        'Megan': ['Meg'],
        'Kelly': ['Kel'],
        'Jessica': ['Jess'],
        'Anna': ['Ann'],
        'Michelle': ['Mikey'],
        'Sarah': ['Sara'],
    }

    def __init__(self, rng: np.random.Generator = None):
        super().__init__(rng)
        # Pre-compute nickname lists for efficiency
        self._nickname_keys = list(self.NICKNAMES.keys())

    def _generate_base_name(self, size: int) -> np.ndarray:
        """Generate base first names from the predefined list."""
        return self.rng.choice(self.FIRST_NAMES, size=size)

    def _apply_typo(self, name: str, typo_prob: float = 0.1) -> str:
        """Apply character-level typos: insertion, deletion, substitution, transposition."""
        if len(name) == 0 or self.rng.random() > typo_prob:
            return name

        # Choose a random typo type
        typo_type = self.rng.choice(['insertion', 'deletion', 'substitution', 'transposition'])
        pos = self.rng.integers(0, len(name))

        if typo_type == 'insertion' and len(name) < 20:
            # Insert a random character
            char = chr(self.rng.integers(97, 123))  # lowercase letter
            return name[:pos] + char + name[pos:]
        elif typo_type == 'deletion' and len(name) > 1:
            # Delete a character
            return name[:pos] + name[pos+1:]
        elif typo_type == 'substitution':
            # Substitute a character
            char = chr(self.rng.integers(97, 123))  # lowercase letter
            return name[:pos] + char + name[pos+1:]
        elif typo_type == 'transposition' and len(name) > 1 and pos < len(name) - 1:
            # Transpose two adjacent characters
            return name[:pos] + name[pos+1] + name[pos] + name[pos+2:]
        else:
            # Fallback to substitution if other operations not possible
            char = chr(self.rng.integers(97, 123))
            return name[:pos] + char + name[pos+1:]

    def _apply_nickname_variation(self, name: str, nickname_prob: float = 0.2) -> str:
        """Apply nickname variation if available."""
        if self.rng.random() > nickname_prob or name not in self.NICKNAMES:
            return name

        nicknames = self.NICKNAMES[name]
        if nicknames:
            return self.rng.choice(nicknames)
        return name

    def _apply_truncation_elongation(self, name: str, trunc_prob: float = 0.15,
                                   elong_prob: float = 0.1) -> str:
        """Apply truncation (cutting off) or elongation (adding characters)."""
        if len(name) == 0:
            return name

        roll = self.rng.random()

        if roll < trunc_prob and len(name) > 2:
            # Truncate: keep first part
            cut_point = self.rng.integers(1, len(name))
            return name[:cut_point]
        elif roll < trunc_prob + elong_prob:
            # Elongate: add random characters at end
            elongation_length = self.rng.integers(1, 4)
            elongation = ''.join(chr(self.rng.integers(97, 123)) for _ in range(elongation_length))
            return name + elongation
        else:
            return name

    def generate_base(self, size: int, config: Dict[str, Any]) -> np.ndarray:
        """
        Generate latent ground-truth first names (no noise, no missingness).
        :param size: Number of entities.
        :param config: Feature configuration.
        :return: NumPy array of shape (size,) with latent first names.
        """
        return self._generate_base_name(size)

    def apply_noise(self, base_values: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Apply noise and missingness to latent base first names.
        Includes character-level typos, nickname variations, and truncation/elongation.
        :param base_values: NumPy array of latent values (shape (n_rows,)).
        :param config: Feature configuration.
        :return: NumPy array of noisy/missing values.
        """
        # Start with base values
        noisy_values = base_values.copy()

        # Get noise parameters from config (with sensible defaults)
        typo_rate = config.get("typo_rate", 0.1)
        nickname_rate = config.get("nickname_rate", 0.2)
        truncation_rate = config.get("truncation_rate", 0.15)
        elongation_rate = config.get("elongation_rate", 0.1)

        # Apply noise to each name
        for i in range(len(noisy_values)):
            name = str(noisy_values[i])

            # Apply transformations in sequence
            name = self._apply_typo(name, typo_rate)
            name = self._apply_nickname_variation(name, nickname_rate)
            name = self._apply_truncation_elongation(name, truncation_rate, elongation_rate)

            noisy_values[i] = name

        # Apply missingness using the helper method
        missing_rate = config.get("missing_rate", 0.0)
        return self._apply_missingness(noisy_values, missing_rate, missing_value=None)


class LastNameGenerator(DataTypeGenerator):
    """Generates last names with noise including typos and truncation/elongation."""

    # Common US last names
    LAST_NAMES = [
        'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
        'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
        'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
        'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
        'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill',
        'Flores', 'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell',
        'Mitchell', 'Carter', 'Roberts', 'Gomez', 'Phillips', 'Evans', 'Turner', 'Diaz',
        'Parker', 'Cruz', 'Edwards', 'Collins', 'Reyes', 'Stewart', 'Morris', 'Morales',
        'Murphy', 'Cook', 'Rogers', 'Gutierrez', 'Ortiz', 'Morgan', 'Cooper', 'Peterson',
        'Bailey', 'Reed', 'Kelly', 'Howard', 'Ramos', 'Kim', 'Cox', 'Ward', 'Richardson',
        'Watson', 'Brooks', 'Chavez', 'Wood', 'James', 'Bennett', 'Gray', 'Mendoza',
        'Ruiz', 'Hughes', 'Price', 'Alvarez', 'Castillo', 'Sanders', 'Patel', 'Myers',
        'Long', 'Ross', 'Foster', 'Jimenez', 'Powell', 'Jenkins', 'Perry', 'Russell',
        'Sullivan', 'Bell', 'Coleman', 'Butler', 'Henderson', 'Barnes', 'Gonzales',
        'Fisher', 'Vasquez', 'Simmons', 'Romero', 'Jordan', 'Patterson', 'Alexander',
        'Hamilton', 'Graham', 'Reynolds', 'Griffin', 'Wallace', 'Moreno', 'West',
        'Cole', 'Hayes', 'Bryant', 'Herrera', 'Gibson', 'Ellis', 'Tran', 'Medina',
        'Aguilera', 'Hawkins', 'Marshall', 'Cornell', 'Tate', 'Fletcher', 'McKinney',
        'Curtis', 'Carney', 'McGee', 'Odom', 'Duke', 'International', 'Engineering',
        'Technology', 'Systems', 'Solutions', 'Group', 'Holdings', 'Enterprises'
    ]

    def __init__(self, rng: np.random.Generator = None):
        super().__init__(rng)

    def _generate_base_name(self, size: int) -> np.ndarray:
        """Generate base last names from the predefined list."""
        return self.rng.choice(self.LAST_NAMES, size=size)

    def _apply_typo(self, name: str, typo_prob: float = 0.1) -> str:
        """Apply character-level typos: insertion, deletion, substitution, transposition."""
        if len(name) == 0 or self.rng.random() > typo_prob:
            return name

        # Choose a random typo type
        typo_type = self.rng.choice(['insertion', 'deletion', 'substitution', 'transposition'])
        pos = self.rng.integers(0, len(name))

        if typo_type == 'insertion' and len(name) < 25:
            # Insert a random character
            char = chr(self.rng.integers(97, 123))  # lowercase letter
            return name[:pos] + char + name[pos:]
        elif typo_type == 'deletion' and len(name) > 1:
            # Delete a character
            return name[:pos] + name[pos+1:]
        elif typo_type == 'substitution':
            # Substitute a character
            char = chr(self.rng.integers(97, 123))  # lowercase letter
            return name[:pos] + char + name[pos+1:]
        elif typo_type == 'transposition' and len(name) > 1 and pos < len(name) - 1:
            # Transpose two adjacent characters
            return name[:pos] + name[pos+1] + name[pos] + name[pos+2:]
        else:
            # Fallback to substitution if other operations not possible
            char = chr(self.rng.integers(97, 123))
            return name[:pos] + char + name[pos+1:]

    def _apply_truncation_elongation(self, name: str, trunc_prob: float = 0.1,
                                   elong_prob: float = 0.05) -> str:
        """Apply truncation (cutting off) or elongation (adding characters)."""
        if len(name) == 0:
            return name

        roll = self.rng.random()

        if roll < trunc_prob and len(name) > 2:
            # Truncate: keep first part
            cut_point = self.rng.integers(1, len(name))
            return name[:cut_point]
        elif roll < trunc_prob + elong_prob:
            # Elongate: add random characters at end
            elongation_length = self.rng.integers(1, 3)
            elongation = ''.join(chr(self.rng.integers(97, 123)) for _ in range(elongation_length))
            return name + elongation
        else:
            return name

    def generate_base(self, size: int, config: Dict[str, Any]) -> np.ndarray:
        """
        Generate latent ground-truth last names (no noise, no missingness).
        :param size: Number of entities.
        :param config: Feature configuration.
        :return: NumPy array of shape (size,) with latent last names.
        """
        return self._generate_base_name(size)

    def apply_noise(self, base_values: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """
        Apply noise and missingness to latent base last names.
        Includes character-level typos and truncation/elongation.
        :param base_values: NumPy array of latent values (shape (n_rows,)).
        :param config: Feature configuration.
        :return: NumPy array of noisy/missing values.
        """
        # Start with base values
        noisy_values = base_values.copy()

        # Get noise parameters from config (with sensible defaults)
        typo_rate = config.get("typo_rate", 0.1)
        truncation_rate = config.get("truncation_rate", 0.1)
        elongation_rate = config.get("elongation_rate", 0.05)

        # Apply noise to each name
        for i in range(len(noisy_values)):
            name = str(noisy_values[i])

            # Apply transformations in sequence
            name = self._apply_typo(name, typo_rate)
            name = self._apply_truncation_elongation(name, truncation_rate, elongation_rate)

            noisy_values[i] = name

        # Apply missingness using the helper method
        missing_rate = config.get("missing_rate", 0.0)
        return self._apply_missingness(noisy_values, missing_rate, missing_value=None)


class DataGeneratorFactory:
    """The Factory Registry that matches configurations to their target type generator classes."""

    _generators = {
        "categorical": CategoricalGenerator(),
        "numerical": NumericalGenerator(),
        "datetime": DateTimeGenerator(),
        "first_name": FirstNameGenerator(),
        "last_name": LastNameGenerator()
    }

    @classmethod
    def get_generator(cls, data_type: str, rng: np.random.Generator = None) -> DataTypeGenerator:
        generator = cls._generators.get(data_type.lower())
        if not generator:
            raise ValueError(f"Unsupported data type generator requested: {data_type}")
        generator.rng = rng or generator.rng
        return generator
