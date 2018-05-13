function add_v1_path()
[dir_this, ~, ~] = fileparts(mfilename('fullpath'));
disp(dir_this)
dir_to_add = fullfile(dir_this, '..', 'V1_ResponseProperties');
disp(dir_to_add)
addpath(dir_to_add);
end

