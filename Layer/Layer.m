classdef Layer < handle
    %LAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
     input 
     output
    end
    
    methods
        
        function FP_output = forward_propagation(obj,input)
            msg = 'NotImplementedError';
            error(msg)
        end
        
        function BP_output = backward_propagation(obj,output_error, learning_rate)
            msg = 'NotImplementedError';
            error(msg)
        end
    end
end

