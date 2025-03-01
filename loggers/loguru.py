import os
from pathlib import Path
from loguru import logger

class LoguruLogger():
    def __init__(self, configs, disable_output_files=False, tqdm_out=False) -> None: # loguru_configs
        super().__init__()
        self.configs = configs
        self.disable_output_files = disable_output_files

        self.logger = logger
        if tqdm_out:
            level = 0 if self.configs.get('debug') else 'INFO'
            self._set_tqdm(level=level)
        self._set_logfiles()
    
    def _set_logfiles(self):
        if self.disable_output_files:
            return

        log_dir = self.configs.get('log_dir')
        if log_dir is not None:
            log_dir = Path(log_dir)
            rank = int(os.environ.get('RANK', 0))
            logfile_prefix = self.configs.get('output_file_prefix', 'loguru')
            
            filename = f'{logfile_prefix}_stdlog{rank}.log'
            self.logger.add(log_dir / filename, level='INFO')

            if self.configs.get('debug'):
                debug_filename = f'{logfile_prefix}_debuglog{rank}.log'
                self.logger.add(log_dir / debug_filename, level=0)
            
            if rank == 0:
                performance_logname = f'{logfile_prefix}_gistlog.log'
                self.logger.add(log_dir / performance_logname, level='SUCCESS')
    
    def _set_tqdm(self, level='INFO'):
        from tqdm import tqdm
        from functools import partial
        self.logger.remove()
        self.logger.add(partial(tqdm.write, end=''), colorize=True, level=level)
