// ====================================================================================
// Math functions
// ====================================================================================

// ====================================================================================
// Interface
var math = {
	
	// Basic operations
	rand    : function(int_roof) {}, // random int from [0, int_roof)
	
	// Scalar functions
	col_max : function(mat, col) {}, // returns max of mat's column # col
	dist    : function(a, b)     {}, // get Euclidean distance between 2 objects
	sum     : function(arr)      {}, // sum of array
	
	// Re-ordering
	shuffle : function(in_array) {}, // shuffle array
	
	// Find in array
	contains : function(arr, val) {}, // true if arr contains val
	index_of : function(arr, val) {}, // get 1st index of val in arr
	last     : function(arr)      {}, // return last elem of array
	max_index: function(arr)      {}, // get max index of array
	
	// Elementwise operations
	add_coords: function(a, b)     {}, // add x & y elements
	add       : function(a, b)     {}, // elementwise addition
	subtract  : function(a, b)     {}, // elementwise subtraction
}

// Basic operations
math.rand = function(int_roof) {
	let result = Math.floor(Math.random() * int_roof);
	return result == int_roof? result-1: result;
}

// Scalar functions
math.col_max = function(mat, col) {
	let out = mat[0][col];
	for (let g = 1; g < mat.length; g++) {
		if (mat[g][col] < out) {continue;}
		out = mat[g][col];
	}
	return out;
}

math.dist = function(a, b) {
	return Phaser.Math.Distance.Between(a.x, a.y, b.x, b.y);
}

math.sum = function(arr) {
	let out = 0;
	for (let g = 0; g < arr.length; g++) {
		out += arr[g];
	}
	return out;
}

// Re-ordering
math.shuffle = function(in_array) {
	len         = in_array.length;
	been_chosen = new Array(len).fill(false);
	out_array   = new Array(len);
	for (let g = 0; g < len; g++) {
		let index;
		do {
			index = math.rand(len);
		} while (been_chosen[index]);
		out_array[g] = in_array[index];
		been_chosen[index] = true;
	}
	return out_array;
}

// Find in array
math.contains = function(arr, val) {
	for (let g = 0; g < arr.length; g++) {
		if (arr[g] == val) {return true;}
	}
	return false;
}

math.index_of = function(arr, val) {
	for (let g = 0; g < arr.length; g++) {
		if (arr[g] == val) {return g;}
	}
	return null;
}

math.last = function(arr) {
	if (arr.length == 0) {return null;}
	return arr[arr.length-1];
}

math.max_index = function(arr) {
	if (arr.length == 0) {return null;}
	let index = 0;
	for (let g = 1; g < arr.length; g++) {
		if (arr[index] < arr[g]) {
			index = g;
		}
	}
	return index;
}

// Elementwise operations
math.add_coords = function(a, b) {
	return {x: a.x + b.x, y: a.y + b.y};
}

math.add = function(a, b) {
	let out = Array(a.length).fill(0);
	for (let g = 0; g < a.length; g++) {
		out[g] = a[g] + b[g];
	}
	return out;
}

math.subtract = function(a, b) {
	let out = Array(a.length).fill(0);
	for (let g = 0; g < a.length; g++) {
		out[g] = a[g] - b[g];
	}
	return out;
}