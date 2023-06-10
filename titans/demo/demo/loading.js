// ====================================================================================
// Loading screen
// ====================================================================================

// ====================================================================================
// Interface
var loading = {
	preload: function() {}, // phaser plugin
	setup  : function() {}, // phaser plugin
	update : function() {}, // phaser plugin
	done   :         false, // whether rest of game can proceed
}

loading.preload = function() {
	
	// create cover images
	loading.background        = [env.add.image({x: env.window.x/2, y: env.window.y/2}, 'cover0')];
	loading.background_order  = [0, 4, 5, 1, 2, 3, 6];
	loading.background_loaded = new Array(loading.background_order.length).fill(false);
	
	// fade in cover as assets load (sort of like a progress bar)
	env.physics.load.on('progress', function (value) {
		for (let fname = 1; fname < loading.background_order.length; fname++) {
			let step_frac  = 1 / (loading.background_order.length + 1);
			let start_frac = fname * step_frac;
			if (!loading.background_loaded[fname] && value >= start_frac) {
				loading.background[fname] = env.add.image({x: env.window.x/2, y: env.window.y/2}, 'cover' + String(fname));
				loading.background_loaded[fname] = true;
				for (let order = 0; order < loading.background_order.length; order++) {
					let index = loading.background_order[order];
					if (order != fname && !loading.background_loaded[index]) {continue;}
					env.to_front(loading.background[index]);
				}
			}
		}
	});
}

loading.setup = function() {
	// skip if restarting
	if (loading.done) {return;}
	
	// bring cover to front
	for (let g = 0; g < 7; g++) {
		env.to_front(loading.background[loading.background_order[g]]);
	}
	
	// make buttons
	loading.play_fs_button = env.add.sprite({x:  540, y: 950}, 'title buttons').setFrame(0).setInteractive();
	loading.play_iw_button = env.add.sprite({x: 1380, y: 950}, 'title buttons').setFrame(1).setInteractive();
	loading.play_fs_button.on('pointerdown', () => {game.phaser.scale.startFullscreen();});
	loading.play_iw_button.on('pointerdown', () => {loading.start();})
}

loading.update = function() {
	// skip if done
	if (loading.done) {return;}
	
	// start if full screen
	if (maximize.scale.isFullscreen) {
		loading.start();
	}
}

// ====================================================================================
// Backend
loading.start = function() {
	for (let g = 0; g < 7; g++) {
		loading.background[g].destroy();
	}
	loading.play_fs_button.destroy();
	loading.play_iw_button.destroy();
	loading.done = true;
}