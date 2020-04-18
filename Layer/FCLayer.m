classdef FCLayer < Layer

    properties       
   
        weights
        bias
    end
    
    methods
        function obj = FCLayer(input_size,output_size)
          
            obj.weights = rand(input_size,output_size) -0.5;
            obj.bias = rand(1,output_size) -0.5;
            obj.input = [];
            obj.output = [];
        end
        
        function output = forward_propagation(obj,input_data)  
            obj.input = input_data;
            obj.output = obj.input*obj.weights +obj.bias;
            output = obj.output;
          
        end
        
        function input_error = backward_propagation(obj,output_error, learning_rate)
            disp(output_error);
            disp(obj.weights);
            input_error = output_error*obj.weights';
            weights_error = obj.input'*output_error;
            % dBias = output_error

            % update parameters
            obj.weights = obj.weights - learning_rate * weights_error;
            obj.bias = obj.bias - learning_rate * output_error;
           
        end
    end
end

