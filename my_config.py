from typing import Any, Dict

import yaml

class BaseConfig(dict):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        if kwargs is not None:
            self.update(kwargs)

    def __setitem__(self, __name: str, __value) -> None:
        if isinstance(__value, dict):
            __value = BaseConfig(**__value)
        super(BaseConfig, self).__setitem__(__name, __value)

    def __getattr__(self, __name: str) -> Any:
        if __name not in self:
            raise AttributeError
        return self[__name]
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__setitem__(__name, __value)

    def update(self, __m, **kwargs):
        for k in __m.keys():
            if hasattr(self, k) and isinstance(self[k], dict) and isinstance(__m[k], dict):
                self[k].update(__m[k])
            else:
                self[k] = __m[k]

    def convert_to_dict(self) -> Dict[str, Any]:
        d = dict()
        for k, v in self.items():
            if isinstance(v, BaseConfig):
                v = v.convert_to_dict()
            d[k] = v
        return d
    
    def load_from_file(self, yml_fp: str) -> None:
        with open(yml_fp, 'r', encoding='UTF-8') as yaml_file:
            self.update(yaml.safe_load(yaml_file))

    def save_to_file(self, yml_fp: str) -> None:
        with open(yml_fp, 'w', encoding='utf-8') as f:
            yaml.dump(self.convert_to_dict(), f)

    def set_config_via_path(self, name_path: str, __value: Any):
        name_path = name_path.split('.')
        cfg = self
        for i, name in enumerate(name_path[:-1]):
            if not isinstance(cfg, BaseConfig):
                raise ValueError(f"invalid path {'.'.join(name_path[:i+1])}")
            if not hasattr(cfg, name):
                cfg[name] = dict()
            cfg = cfg[name]
        cfg[name_path[-1]] = __value

    def has_attr_via_path(self, name_path: str):
        name_path = name_path.split('.')
        cfg = self
        for name in name_path:
            try:
                cfg = cfg[name]
            except Exception as e:
                return False
        return True
    
    def get_attr_via_path(self, name_path: str):
        name_path = name_path.split('.')
        cfg = self
        for i, name in enumerate(name_path):
            try:
                cfg = cfg[name]
            except Exception as e:
                return ValueError(f"Path {'.'.join(name_path[:i+1])} does not exist.")
        return cfg


if __name__ == '__main__':
    import os

    conf = BaseConfig()
    conf.a = 'aa'
    conf.l = [1,2,3]
    conf.d = {'b': 2}
    print(conf)
    conf.save_to_file('test.yml')

    conf2 = BaseConfig()
    conf2.set_config_via_path('l', 3)
    print(conf2)
    print(conf2.has_attr_via_path('d.b'))
    conf2.load_from_file('test.yml')
    print(conf2)
    print(conf2.has_attr_via_path('d.b'))

    os.remove('test.yml')