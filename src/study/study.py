from pathlib import Path
from typing import Optional, List
from datetime import datetime
import dataclasses, pickle


from collection.collection import DatasetCollection
from reader.reader import BaseFileReader
from experiment.experiment import Experiment
from study.design import StudyDesign, ExperimentSpec
from results.containers import MultiDomainSolution, StudySolution, StudySolutionBuilder

class Study:
    """
    Executes and manages a complete research study.
    
    Handles:
    - Running experiments across all configs and seeds
    - Collecting results into StudySolution format
    - Saving/loading results
    
    Example:
        study = Study(collection, reader)
        design = StudyDesign(
            name='bearing_study',
            experiment_specs=(spec1, spec2),
            seeds=(42, 123, 456)
        )
        results = study.run(design)
        study.save('bearing_study', results, design)
    """
    
    def __init__(
        self,
        collection: DatasetCollection,
        reader: BaseFileReader,
        results_dir: Path = Path('./results')
    ):
        self._collection = collection
        self._reader = reader
        self._results_dir = results_dir
        self._results_dir.mkdir(exist_ok=True, parents=True)
    
    def run(self, design: StudyDesign, verbose: bool = True) -> StudySolution:
        """
        Execute complete study.
        
        Args:
            design: StudyDesign specification
            verbose: Whether to print progress
            
        Returns:
            StudySolution containing all results
        """
        builder = StudySolutionBuilder(design.name)
        builder.set_metadata('design_description', design.description)
        builder.set_metadata('design_metadata', design.metadata)
        
        total_specs = len(design.experiment_specs)
        
        for spec_idx, spec in enumerate(design.experiment_specs):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Experiment {spec_idx + 1}/{total_specs}: {spec.name}")
                print(f"Task: {spec.task.target}")
                print(f"Model: {spec.config.model_class.__name__}")
                print(f"Processor: {spec.processor_name}")
                print(f"Domains: {spec.num_domains}")
                print(f"Seeds: {design.seeds}")
                print(f"{'='*70}")
            
            # Run across all seeds for this spec
            multi_domain_solutions = self._run_spec_multi_seed(
                spec=spec,
                seeds=design.seeds,
                verbose=verbose
            )
            
            # Add all solutions to builder
            for mds in multi_domain_solutions:
                builder.add_multi_domain_solution(mds)
        
        return builder.build()
    
    def _run_spec_multi_seed(
        self,
        spec: ExperimentSpec,
        seeds: tuple[int, ...],
        verbose: bool = True
    ) -> List[MultiDomainSolution]:
        """Run one experiment spec across multiple seeds."""
        
        multi_domain_solutions = []
        
        for seed_idx, seed in enumerate(seeds):
            if verbose:
                print(f"\n--- Seed {seed_idx + 1}/{len(seeds)}: {seed} ---")
            
            # Create config with this seed
            config = dataclasses.replace(spec.config, random_seed=seed)
            
            # Create experiment instance (processor created from config inside Experiment)
            experiment = Experiment(
                collection=self._collection,
                reader=self._reader,
                config=config
            )
            
            # Run pairwise evaluation
            mds = experiment.run_pairwise(spec.task, spec.filter_combinations)
            multi_domain_solutions.append(mds)
        
        return multi_domain_solutions
    
    def run_and_save(
        self, 
        design: StudyDesign, 
        verbose: bool = True
    ) -> tuple[StudySolution, Path]:
        """
        Run study and save results.
        
        Returns:
            Tuple of (StudySolution, save_path)
        """
        results = self.run(design, verbose)
        save_path = self.save(design.name, results, design)
        return results, save_path
    
    def save(
        self, 
        name: str, 
        results: StudySolution, 
        design: Optional[StudyDesign] = None
    ) -> Path:
        """
        Save results to disk.
        
        Creates directory with:
        - results.pkl: StudySolution
        - design.pkl: StudyDesign (if provided)
        - metadata.txt: Human-readable summary
        
        Args:
            name: Base name for save directory
            results: StudySolution to save
            design: Optional StudyDesign for reproducibility
            
        Returns:
            Path to save directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self._results_dir / f"{name}_{timestamp}"
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save results
        with open(save_dir / "results.pkl", 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save design
        if design:
            with open(save_dir / "design.pkl", 'wb') as f:
                pickle.dump(design, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save human-readable summary
        summary_lines = [
            f"Study: {results.study_name}",
            f"Timestamp: {results.timestamp}",
            f"Configs: {results.config_names}",
            f"Seeds: {results.get_all_seeds()}",
            "",
            results.summary()
        ]
        with open(save_dir / "metadata.txt", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"\nResults saved to {save_dir}")
        return save_dir
    
    @staticmethod
    def load(path: Path) -> tuple[StudySolution, Optional[StudyDesign]]:
        """
        Load results from disk.
        
        Args:
            path: Path to save directory
            
        Returns:
            Tuple of (StudySolution, StudyDesign or None)
        """
        path = Path(path)
        
        with open(path / "results.pkl", 'rb') as f:
            results = pickle.load(f)
        
        design = None
        design_path = path / "design.pkl"
        if design_path.exists():
            with open(design_path, 'rb') as f:
                design = pickle.load(f)
        
        return results, design
    
    def list_saved(self) -> List[Path]:
        """List all saved study directories."""
        return sorted([
            p for p in self._results_dir.iterdir()
            if p.is_dir() and (p / "results.pkl").exists()
        ])
