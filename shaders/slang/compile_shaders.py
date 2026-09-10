import re
import os
import sys
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

STAGE_MAPPINGS = {
    "vertex": {"stage_suffix": "vert", "entry": "vertexMain"},
    "fragment": {"stage_suffix": "frag", "entry": "fragmentMain"},
    "compute": {"stage_suffix": "comp", "entry": "computeMain"},
    "geometry": {"stage_suffix": "geom", "entry": "geometryMain"},
}

COMPILE_TARGETS = {
    "spirv": {"suffix": "spv"},
}
COMPILE_PROFILE = "glsl_450"
COMPILE_OPTIONS = [
    "-fvk-use-scalar-layout",
    "-matrix-layout-column-major",
]

HASH_FILE = ".shader_cached.json"

SPIRV_VALIDATOR = "spirv-val"
SPIRV_VALIDATOR_OPTIONS = [
    "--target-env", "vulkan1.3",
    "--scalar-block-layout",
]

class BuildDescription:
    def __init__(self, json_path="shaders.json"):
        self.root_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.include_dirs = [self.root_dir]
        self.output_dir = os.path.join(self.root_dir, "bin_shaders")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if 'include_dirs' in config:
                self.include_dirs = [os.path.abspath(os.path.join(self.root_dir, d)) for d in config['include_dirs']]
            
            if 'output_dir' in config:
                self.output_dir = os.path.abspath(os.path.join(self.root_dir, config['output_dir']))

        except FileNotFoundError:
            print(f"Warning: Build configuration file '{json_path}' not found. Using default paths.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{json_path}'. Using default paths.")


def get_file_hash(filepath, desc: BuildDescription):
    import_pattern = re.compile(r'import\s+([a-zA-Z_][\w.]*);')
    hasher = hashlib.sha256()

    included_files = set()
    
    def find_module_file(module_name: str):
        path_name = module_name.replace('.', os.sep)

        for inc_dir in desc.include_dirs:
            potential_path_1 = os.path.join(inc_dir, path_name + '.slang')
            if os.path.exists(potential_path_1):
                return os.path.abspath(potential_path_1)

            potential_path_2 = os.path.join(inc_dir, path_name, os.path.basename(path_name) + '.slang')
            if os.path.exists(potential_path_2):
                return os.path.abspath(potential_path_2)

        return None

    def hash_file_recursive(path):
        abs_path = os.path.abspath(path) 
        
        if abs_path in included_files:
            return True

        included_files.add(abs_path)

        try:
            with open(abs_path, 'rb') as f:
                content = f.read()
                hasher.update(content)

                content_str = content.decode('utf-8', errors='ignore')

                for match in import_pattern.finditer(content_str):
                    module_name = match.group(1)
                    found_path = find_module_file(module_name)
                    
                    if found_path:
                        if not hash_file_recursive(found_path):
                            return False
                    
        except FileNotFoundError:
            return False

        return True

    if not hash_file_recursive(filepath):
        return None
        
    return hasher.hexdigest()

def load_hashes(desc: BuildDescription):
    hash_path = os.path.join(desc.output_dir, HASH_FILE)
    if os.path.exists(hash_path):
        try:
            with open(hash_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_hashes(hashes, desc: BuildDescription):
    os.makedirs(desc.output_dir, exist_ok=True)
    hash_path = os.path.join(desc.output_dir, HASH_FILE)
    try:
        with open(hash_path, 'w', encoding='utf-8') as f:
            json.dump(hashes, f, indent=4)
    except IOError:
        pass

def parse_metadata_to_stages(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return []

    match = re.search(r'//\s*\[moe\s*\((.*?)\)\]', content)

    if not match:
        return []

    stages_str = match.group(1).replace('"', '').replace("'", '').strip()
    stage_names = [s.strip() for s in stages_str.split(',')]
    
    return [s for s in stage_names if s in STAGE_MAPPINGS]

def enumerate_files(desc: BuildDescription):
    files = []
    for inc_dir in desc.include_dirs:
        for root, _, filenames in os.walk(inc_dir):
            for filename in filenames:
                if filename.endswith('.slang'):
                    filepath = os.path.abspath(os.path.join(root, filename))
                    if filepath not in files:
                        files.append(filepath)
    return files

def generate_output_path(filepath, stage, target, desc: BuildDescription):
    relative_path = None

    for inc_dir in desc.include_dirs:
        abs_inc_dir = os.path.abspath(inc_dir)

        if filepath.startswith(abs_inc_dir):
            relative_path = os.path.relpath(filepath, abs_inc_dir)
            break

    if relative_path is None:
        relative_path = os.path.relpath(filepath, desc.root_dir)

    output_base_dir = os.path.join(desc.output_dir, os.path.dirname(relative_path))
    os.makedirs(output_base_dir, exist_ok=True)

    output_filename = generate_output_filename(filepath, stage, target)
    return os.path.join(output_base_dir, output_filename)

def generate_output_filename(filepath, stage, target):
    stage_suffix = STAGE_MAPPINGS[stage]["stage_suffix"]
    file_suffix = COMPILE_TARGETS[target]["suffix"]
    base_name = os.path.basename(filepath)
    return f"{os.path.splitext(base_name)[0]}.{stage_suffix}.{file_suffix}"

def generate_compile_command(filepath, stage, target, output_path, desc: BuildDescription):
    stage_info = STAGE_MAPPINGS[stage]
    entry_point = stage_info["entry"]
    
    include_options = [f"-I {d}" for d in desc.include_dirs]
    
    cmd_parts = [
        "slangc",
        filepath,
        f"-target {target}",
        f"-profile {COMPILE_PROFILE}",
        f"-stage {stage}",
    ] + include_options + [
        f"-entry {entry_point}",
    ] + COMPILE_OPTIONS + [
        f"-o {output_path}",
    ]
    
    return " ".join(cmd_parts)

def execute_command_with_hash(filepath, stage, target, current_hashes, new_hashes, desc: BuildDescription):
    key = f"{filepath}::{stage}::{target}"
    
    file_hash = get_file_hash(filepath, desc)
    output_path = generate_output_path(filepath, stage, target, desc)

    if file_hash is None:
        return f"ERROR: File not found or included file missing: {filepath}"

    if current_hashes.get(key) == file_hash and os.path.exists(output_path):
        new_hashes[key] = file_hash
        return f"SKIP: {os.path.relpath(output_path, desc.output_dir)} (Hash Match)"

    command = generate_compile_command(filepath, stage, target, output_path, desc)
    status = os.system(command)
    
    if status != 0:
        return f"ERROR: Command failed with status {status} (Output: {os.path.relpath(output_path, desc.output_dir)}): {command}"

    new_hashes[key] = file_hash
    return f"BUILD: {os.path.relpath(output_path, desc.output_dir)}"

def compile_all_parallel(desc: BuildDescription):
    current_hashes = load_hashes(desc)
    new_hashes = {}
    compile_tasks = []
    
    for filepath in enumerate_files(desc):
        stages = parse_metadata_to_stages(filepath)
        
        for stage in stages:
            for target in COMPILE_TARGETS:
                task_args = (filepath, stage, target, current_hashes, new_hashes, desc)
                compile_tasks.append(task_args)
                
    if not compile_tasks:
        print("No .slang files found or no stages defined.")
        return

    print(f"Executing {len(compile_tasks)} compilation tasks concurrently to {desc.output_dir}...")

    max_workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda args: execute_command_with_hash(*args), 
            compile_tasks
        ))

    save_hashes(new_hashes, desc)
    
    print("-" * 50)
    failed_count = 0
    for result in results:
        print(result)
        if result.startswith("ERROR:"):
            failed_count += 1

    print("-" * 50)
    if failed_count > 0:
        print(f"Shader compilation finished with {failed_count} failures.")
        sys.exit(1)
    else:
        print("Shader compilation finished successfully.")

def clean_build(desc: BuildDescription):
    cleaned_files = 0
    
    if os.path.exists(desc.output_dir):
        for root, _, filenames in os.walk(desc.output_dir, topdown=False):
            for filename in filenames:
                if any(filename.endswith(f".{info['suffix']}") for info in COMPILE_TARGETS.values()) or filename == HASH_FILE:
                    os.remove(os.path.join(root, filename))
                    cleaned_files += 1

            if not os.listdir(root):
                os.rmdir(root)
                
    print(f"Clean successful. Removed {cleaned_files} files from {desc.output_dir}.")

def find_spv_files(desc: BuildDescription):
    spv_files = []
    if os.path.exists(desc.output_dir):
        for root, _, filenames in os.walk(desc.output_dir):
            for filename in filenames:
                if filename.endswith(".spv"):
                    spv_files.append(os.path.join(root, filename))
    return spv_files

def validate_spirv_file(spv_file):
    command = " ".join([
        SPIRV_VALIDATOR,
        *SPIRV_VALIDATOR_OPTIONS,
        spv_file
    ])
    status = os.system(command)
    
    if status != 0:
        return f"VALIDATION FAILED ({os.path.relpath(spv_file)}): Check console output."
    return f"VALIDATION SUCCESS: {os.path.relpath(spv_file)}"

def validate_all_parallel(desc: BuildDescription):
    spv_files = find_spv_files(desc)
    
    if not spv_files:
        print("No .spv files found to validate. Run 'build' first.")
        return

    print(f"Validating {len(spv_files)} SPIR-V files concurrently...")

    max_workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(validate_spirv_file, spv_files))
    
    print("-" * 50)
    failed_count = 0
    for result in results:
        print(result)
        if "FAILED" in result:
            failed_count += 1
            
    print("-" * 50)
    if failed_count > 0:
        print(f"Validation finished with {failed_count} failures.")
        sys.exit(1)
    else:
        print("All SPIR-V files passed validation.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <command> [config_file.json]")
        print("Commands: build, clean, rebuild, validate")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    config_file = sys.argv[2] if len(sys.argv) > 2 else "shaders.json"
    desc = BuildDescription(config_file)
    
    if command == "build":
        compile_all_parallel(desc)
    elif command == "clean":
        clean_build(desc)
    elif command == "rebuild":
        clean_build(desc)
        compile_all_parallel(desc)
    elif command == "validate":
        validate_all_parallel(desc)
    else:
        print(f"Unknown command: {command}")
        print("Commands: build, clean, rebuild, validate")
        sys.exit(1)