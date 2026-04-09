from collections import defaultdict
from pathlib import Path
from .task import Task
from .dataset_plan import DatasetPlan, SampleGroup
from .metadata import Metadata


import yaml
import itertools


class DatasetCollection:
    """Load a collection YAML and provide structured access to files and samples."""
    @staticmethod
    def check_attribute_value(attribute:int, attribute_name:str):
        if (attribute < 0 or attribute > 9):
            raise ValueError(f'{attribute_name} {attribute} must be in range (0,9)')
    def __init__(self, yaml_path: Path|str, check_files: bool = True):
        self._yaml_path = Path(yaml_path)
        if not self._yaml_path.exists():
            raise ValueError(f'Collection configuration file {self._yaml_path} does not exist!')

        with open(yaml_path, "r") as f:
            self._cfg = yaml.safe_load(f)

        if check_files:
            not_found = self._check_file_exists()
            if not_found:
                not_found_str = ', '.join(map(str, not_found))
                raise ValueError(f'Some of the files not found: {not_found_str}! Please fix yaml config!')

        self._samples = defaultdict(list)
        self._generate_samples()

        reader_path = self._cfg.get('reader')
        if reader_path:
            from ..reader.config import ReaderConfig
            from ..reader.reader import UniversalFileReader
            self._reader = UniversalFileReader(ReaderConfig.from_yaml(reader_path))
        else:
            self._reader = None

        from .channels import parse_all_channels
        raw_channels = self._cfg.get('channels', {})
        self._channels = parse_all_channels(raw_channels) if raw_channels else {}

        if self._reader and self._channels:
            reader_ch_names = set(self._reader._cfg.channels.keys())
            for ch_name, ch_cfg in self._channels.items():
                if hasattr(ch_cfg, 'reader_channel'):
                    if ch_cfg.reader_channel not in reader_ch_names:
                        raise ValueError(
                            f"Channel '{ch_name}' references reader_channel "
                            f"'{ch_cfg.reader_channel}' which doesn't exist in reader. "
                            f"Available: {reader_ch_names}"
                        )

        if self._samples and self._channels:
            first_code = next(iter(self._samples.keys()))
            test_meta = Metadata(self.code_description(first_code))
            for ch_name, ch_cfg in self._channels.items():
                if hasattr(ch_cfg, 'metadata_path'):
                    try:
                        val = test_meta
                        for part in ch_cfg.metadata_path.split('.'):
                            val = val[part]
                    except (KeyError, TypeError):
                        raise ValueError(
                            f"Channel '{ch_name}' metadata_path "
                            f"'{ch_cfg.metadata_path}' cannot resolve in metadata"
                        )

    def _check_file_exists(self) -> tuple[str, ...]:
        """Return (item, path) pairs for every expected file that is missing."""
        not_found = []
        for item, vals in self.files.items():
            realizations = 1

            subdirectory = ''

            if isinstance(vals, dict):
                realizations = vals.get('realizations', 1)

                subdirectory = vals.get('subdirectory', '')

            base = self.dirname / subdirectory / f'{item}.{self.filetype}'

            fnames = []
            
            if realizations > 1:
                fnames.extend(
                    str(base.with_name(f'{item}_{i}.{self.filetype}'))
                    for i in range(1, realizations + 1)
                )
            else:
                fnames.append(str(base))
            for fname in fnames:
                if not Path(fname).exists():
                    not_found.append((item, fname))
        return tuple(not_found)
    
    def _alias_to_code(self, field: str, alias: str) -> int:
        for code, desc in self.header[field].items():
            if isinstance(desc, dict) and desc.get('alias') == alias:
                return code
        raise ValueError(f"Alias '{alias}' not found for filter '{field}'.")

    def _generate_samples(self):
        schema_fields = set(self.schema.keys())
        for fname, entry in self.files.items():
            filter_values = {}
            for field in schema_fields:
                raw = entry[field]
                if isinstance(raw, str):
                    filter_values[field] = self._alias_to_code(field, raw)
                else:
                    filter_values[field] = raw
            code = self.construct_code(**filter_values)
            self._samples[code].append(fname)
    
    def _validate_task(self, task:Task):
        target = task.target
        domain_factors = task.domain_factors
        defaults = task.defaults
        classes = task.classes

        if target not in self.header:
            raise ValueError(f'Unknown target {target} specified in task!')
        
        if defaults.fixed.keys() - self.header.keys():
            missing = defaults.fixed.keys() - self.header.keys()
            missing_str = ', '.join(map(str, sorted(missing)))
            raise ValueError(f'Arguments {missing_str} in depends not present in dataset collection definition')
        
        if defaults.resolve.keys() - self.header.keys():
            missing = defaults.resolve.keys() - self.header.keys()
            missing_str = ', '.join(map(str, sorted(missing)))
            raise ValueError(f'Arguments {missing_str} in default resolving dictionary not present in dataset collection definition')
        
        if classes.keys() - self.header[target].keys():
            missing = classes.keys() - self.header[target].keys()
            missing_str = ', '.join(map(str, sorted(missing)))
            raise ValueError(f'Arguments {missing_str} in default resolving dictionary not present in dataset collection definition')
        
        filters = {target}
        filters.update(defaults.fixed.keys())
        filters.update(defaults.resolve.keys())

        if filters != self.header.keys():
            missing = self.header.keys() - filters
            missing_str = ', '.join(missing)
            raise ValueError (f'Task must be complete. Filters {missing_str} are missing.')
        
        for cls, cls_rule in classes.items():
            if cls_rule.resolve.keys() - self.header.keys():
                missing = cls_rule.resolve.keys() - self.header.keys()
                missing_str = ', '.join(map(str, sorted(missing)))
                raise ValueError(f'Arguments {missing_str} in classes {cls} specific resolving dictionary not present in dataset collection definition')

    @property
    def name(self) -> str:
        return self._cfg['name']

    @property
    def header(self) -> dict:
        return self._cfg['header']

    @property
    def files(self) -> dict:
        return self._cfg['files']

    @property
    def samples(self) -> dict:
        return self._samples

    @property
    def filetype(self) -> str:
        return self._cfg['filetype']

    @property
    def dirname(self) -> Path:
        return Path(self._cfg['dirname'])

    @property
    def schema(self) -> dict:
        return self._cfg['code_schema']
    
    def construct_code(self, **filters) -> int:
        """Compute the composite integer code for a set of filter values."""
        code = 0

        for field, multiplier in self.schema.items():
            field_value = filters[field]
            self.check_attribute_value(field_value, field)
            code += field_value * multiplier

        return code
    
    def code_description(self, code:int)->dict:
        description = {}
        for field, multiplier in self.schema.items():
            field_value = int((code // multiplier) % 10)
            desc = self.header[field][field_value]
            if isinstance(desc, dict):
                description[field] = {k: v for k, v in desc.items() if k != 'alias'}
            else:
                description[field] = {'name': desc, 'value': desc}
        return description

    @property
    def reader(self):
        return self._reader

    @property
    def channels(self) -> dict:
        return self._channels

    def get_filenames_from_code(self, code) -> tuple[str, ...]:
        return tuple(self.samples[code])
    
    def add_realizations_to_the_code(self, code:int, realizations:int)->int:
        from math import log10, floor
        return realizations * 10 ** (floor(log10(code)) + 1) + code


    def construct_code_label(self, code):
        return ''
    
    def construct_dataset_plan(self, task:Task, **filters)->DatasetPlan:
        target = task.target
        defaults = task.defaults
        classes = task.classes
        interactions = task.interactions
        class_interactions = task.class_interactions

        self._validate_task(task)

        if filters.keys() - self.header.keys():
            unknown = filters.keys() - self.header.keys()
            unknown_str = ', '.join(unknown)
            raise ValueError(f'Unknown filters: {unknown_str} specified as keyword arguments!')
        
        sample_groups = {}

        for cls_label in self.header[target]:
            full_resolved = {}

            full_resolved.update(defaults.fixed)
            full_resolved.update(defaults.resolve)

            full_resolved.update(filters)

            if cls_label in classes:
                full_resolved.update(classes[cls_label].fixed)

                full_resolved.update(classes[cls_label].resolve)

            
            keys = list(full_resolved.keys())

            values_list = [
                v if isinstance(v, list) else [v]
                for v in full_resolved.values()
            ]

            codes = {}

            metadata = {}

            for combination in itertools.product(*values_list):
                resolved = dict(zip(keys, combination))
                if class_interactions and cls_label in class_interactions:
                    if not class_interactions[cls_label].is_satisfied_by(resolved):
                        continue
                if interactions:
                    if not interactions.is_satisfied_by(resolved):
                        continue
                
                code = self.construct_code(**{task.target:cls_label}, **resolved)

                if code not in self.samples:
                    raise ValueError(f'Combination provided with filters {filters} not in the collection')

                fnames = []

                for fname in self.samples[code]:
                    file_entry = self.files[fname]
                    
                    realizations = 1

                    subdirectory = ''

                    if isinstance(file_entry, dict):
                        realizations = file_entry.get('realizations', 1)

                        subdirectory = file_entry.get('subdirectory', '')


                    base = self.dirname / subdirectory / f'{fname}.{self.filetype}'

                    if realizations > 1:
                        fnames.extend(
                            str(base.with_name(f'{fname}_{i}.{self.filetype}'))
                            for i in range(1, realizations + 1)
                        )
                    else:
                        fnames.append(str(base))

                codes[code] = fnames


                metadata[code] = Metadata(self.code_description(code))


            cls_desc = self.header[target][cls_label]
            cls_label_str = cls_desc['name'] if isinstance(cls_desc, dict) else cls_desc
            sample_groups[cls_label_str] = SampleGroup(
                codes=codes,
                metadata=metadata
            )
        #if self.name == 'cwru':
            #print(all([bool(_.codes) for _ in sample_groups.values()]), sample_groups)
        #if not all([bool(_.codes) for _ in sample_groups.values()]):
        #    raise RuntimeError('One of the datasets empty')

        return DatasetPlan(
            self.name,
            task.label(**filters),
            sample_groups
        )
    def get_all_filter_values(self, filter:str)->list[str]:
        return list(self.header[filter].keys())
    
    def get_filter_value_from_description(self, filter: str, description) -> int:
        for code, desc in self.header[filter].items():
            if isinstance(desc, dict):
                if desc.get('alias') == description or desc.get('name') == description:
                    return code
            elif desc == description:
                return code
        raise ValueError(f"Filter '{filter}' value '{description}' not found in header.")
    
    def create_filters_combinations_from_depends(self, depends, **excludes) -> tuple[dict, ...]:

        if set(depends) - self.header.keys():
            missing = set(depends) - self.header.keys()
            missing_str = ', '.join(map(str, missing))
            raise ValueError(f'Factors {missing_str} from depends missing in header definition')
        
       
        filters = {}

        for flt in depends:
            flt_vals = self.header[flt].keys()

            exclude = excludes.get(flt, None)

            if exclude is not None:

                if isinstance(exclude, int):
                    exclude = [exclude]


                flt_vals = [v for v in flt_vals if v not in exclude]

            filters[flt] = tuple(flt_vals)

        filter_combinations = []

        for filter_comb in itertools.product(*filters.values()):
            filter_comb_dict = {flt:val for flt,val in zip(filters.keys(), filter_comb)}
            
            filter_combinations.append(filter_comb_dict)      

        return tuple(filter_combinations)
    
    def _check_filter_produces_complete_plan(self, task: Task, filters: dict) -> bool:
        """
        Check if a single filter combination produces a complete dataset plan.
        
        This is a lightweight check that doesn't construct the full plan.
        
        Args:
            task: Task definition
            filters: Single filter combination dict
            
        Returns:
            True if filter would produce complete plan, False otherwise
        """
        target = task.target
        defaults = task.defaults
        classes = task.classes
        interactions = task.interactions
        class_interactions = task.class_interactions
        
        # Check each class
        for cls_label in self.header[target]:
            # Build full resolved dict for this class
            full_resolved = {}
            full_resolved.update(defaults.fixed)
            full_resolved.update(defaults.resolve)
            full_resolved.update(filters)
            
            if cls_label in classes:
                full_resolved.update(classes[cls_label].fixed)
                full_resolved.update(classes[cls_label].resolve)
            
            # Expand resolve lists to check all combinations
            keys = list(full_resolved.keys())
            values_list = [
                v if isinstance(v, list) else [v]
                for v in full_resolved.values()
            ]
            
            # Check if at least one combination exists for this class
            has_valid_combination = False
            
            for combination in itertools.product(*values_list):
                resolved = dict(zip(keys, combination))
                
                # Check class interactions
                if class_interactions and cls_label in class_interactions:
                    if not class_interactions[cls_label].is_satisfied_by(resolved):
                        continue
                
                # Check global interactions
                if interactions:
                    if not interactions.is_satisfied_by(resolved):
                        continue
                
                # Try to construct code - if it exists, this class has data
                
                code = self.construct_code(**{task.target: cls_label}, **resolved)
                if code in self.samples and self.samples[code]:
                    has_valid_combination = True
                    break  # Found at least one valid code for this class
                
            
            if not has_valid_combination:
                # This class has no valid samples for this filter
                return False
        
        return True
    
    def validate_filters(
        self, 
        task: Task, 
        filters: dict | list[dict]
    ) -> dict | list[dict] | None:
        """
        Validate filter(s) against a task.
        
        Checks if the filter combination(s) would produce complete dataset plans
        (all classes have at least one sample).
        
        Args:
            task: Task definition
            filters: Single filter dict or list of filter dicts
            
        Returns:
            - If single filter: returns filter if valid, None if invalid
            - If list of filters: returns list of valid filters, None if all invalid
            
        Example:
            # Single filter
            valid = collection.validate_filters(task, {'fault_size': 1, 'condition': 0})
            if valid:
                plan = collection.construct_dataset_plan(task, **valid)
            
            # Multiple filters  
            all_filters = collection.create_filters_combinations_from_depends(...)
            valid_filters = collection.validate_filters(task, list(all_filters))
            if valid_filters:
                for flt in valid_filters:
                    plan = collection.construct_dataset_plan(task, **flt)
        """
        if isinstance(filters, dict):
            # Single filter
            if self._check_filter_produces_complete_plan(task, filters):
                return filters
            return None
        
        elif isinstance(filters, (list, tuple)):
            # List of filters
            valid = [
                flt for flt in filters 
                if self._check_filter_produces_complete_plan(task, flt)
            ]
            return valid if valid else None
        
        else:
            raise TypeError(f"filters must be dict or list[dict], got {type(filters)}")
    
    def create_valid_filter_combinations(
        self,
        task: Task,
        depends: tuple[str, ...],
        **excludes
    ) -> tuple[dict, ...]:
        """
        Create filter combinations that are guaranteed to produce complete plans.
        
        Combines create_filters_combinations_from_depends with validate_filters.
        
        Args:
            task: Task to validate against
            depends: Tuple of factor names to vary
            interactions: Optional interaction constraints
            **excludes: Factor values to exclude
            
        Returns:
            Tuple of valid filter dicts (may be empty)
            
        Example:
            valid_filters = collection.create_valid_filter_combinations(
                task,
                ('fault_size', 'bearing_position', 'condition'),
                fault_size=[0, 4]  # exclude these
            )
            
            for flt in valid_filters:
                plan = collection.construct_dataset_plan(task, **flt)
                # Guaranteed to be complete
        """
        all_filters = self.create_filters_combinations_from_depends(
            depends, **excludes
        )
        
        valid = self.validate_filters(task, list(all_filters))
        
        return tuple(valid) if valid else ()
    