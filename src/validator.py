class ResultValidator:
    """Validate and normalize results"""
    
    def __init__(self, config):
        self.config = config
        self.auto_normalize = config.cfg['validation']['auto_normalize']
    
    def validate(self, result: dict, duration: float) -> tuple:
        """Validate result constraints"""
        errors = []
        
        # Check attractor sum
        attractor_total = sum(result.get('attractor_seconds', {}).values())
        if abs(attractor_total - duration) > 0.5:
            errors.append(f"Attractor sum ({attractor_total:.1f}) != duration ({duration:.1f})")
        
        # Check narrative frames per modality
        for modality in ['visual', 'audio', 'text']:
            key = f'narrative_{modality}'
            if key in result:
                total = sum(result[key].values())
                if abs(total - duration) > 0.5:
                    errors.append(f"{modality} frame sum ({total:.1f}) != duration ({duration:.1f})")
        
        return len(errors) == 0, errors
    
    def normalize(self, result: dict, duration: float) -> dict:
        """Normalize results to match duration exactly"""
        
        # Normalize attractors
        attractor_total = sum(result.get('attractor_seconds', {}).values())
        if attractor_total > 0:
            factor = duration / attractor_total
            result['attractor_seconds'] = {
                k: v * factor for k, v in result['attractor_seconds'].items()
            }
        
        # Normalize narrative frames
        for modality in ['visual', 'audio', 'text']:
            key = f'narrative_{modality}'
            if key in result:
                total = sum(result[key].values())
                if total > 0:
                    factor = duration / total
                    result[key] = {k: v * factor for k, v in result[key].items()}
        
        return result