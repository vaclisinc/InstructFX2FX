var fs = 48000;

/**
 * 40-band graphic equalizer AudioNode. <br>
 * Center frequencies are logarithmically spaced between 20 Hz and 20 KHz
 * @class
 * @name Equalizer
 * @param {AudioContext} context - The Web Audio context
 * @param {Object} opts - Can contain values for eq.param.curve and
 * eq.param.range
 */
function Equalizer (context, opts) {
	this.input = context.createGain();
	this.output = context.createGain();

	this.normalize = true;
	
	opts = opts || {}

	this.param = new Object();
	this.param.curve = opts.curve;
	this.param.range = opts.range;
	if (this.param.range == undefined) {
		this.param.range = 1;
	}
	this.filters = [];
	var freqs = [20, 50, 83, 120, 161, 208, 259, 318, 383, 455, 537, 628, 729, 843, 971, 1114, 1273, 1452, 1652, 1875, 2126, 2406, 2719, 3070, 3462, 3901, 4392, 4941, 5556, 6244, 7014, 7875, 8839, 9917, 11124, 12474, 13984, 15675, 17566, 19682];

	for (i = 0; i < 40; i++) {
		this.filters.push(context.createBiquadFilter());
		this.filters[i].type = "peaking";
		this.filters[i].frequency.value = freqs[i];
		this.filters[i].Q.value = 4.31;

		this.filters[i].gain.value = this.param.curve[i];
		if (i > 0) {
			this.filters[i-1].connect(this.filters[i]);
		}
	}

	this.input.connect(this.filters[0]);
	this.filters[this.filters.length - 1].connect(this.output);	
}

Equalizer.prototype = Object.create(null, {
  connect: {
    value: function(dest) {
      this.output.connect(dest.input ? dest.input : dest);
    }
  },

  disconnect: {
	  value: function () {
		  this.output.disconnect();
	  }
  },

  /**
   * The center frequencies of the 40 equalizer bands. (read only)
   * @name Equalizer~cfreqs
   * @member
   * @type Array
   */  
  cfreqs: {
	  get: function () {
		var cfreqs = new Array(40);
		for (var i = 0; i < 40; i++) {
		  cfreqs[i] = this.filters[i].frequency.value;
		}
		return cfreqs;  
	  }
  },

  /**
   * The EQ curve. <br>
   * An array of 40 numbers corresponding to the gain values in
   * dB for each filter in the Equalizer
   * @name Equalizer~curve
   * @member
   * @type Array
   */
  curve: {
    get: function() { return this.param.curve; },
    set: function(value) {
      if (value.length != 40) {
        return -1;
      }
      this.param.curve = [];
      max_el = 0;
      min_el = 0;
      for (i = 0; i < 40; i++) {
        if (value[i] > max_el) {
          max_el = value[i];
        }
        if (value[i] < min_el) {
          min_el = value[i];
        }
      }

      for (i = 0; i < 40; i++) {
        if (this.normalize) {
          dat = value[i] - min_el;
          if (max_el != min_el) {
			  dat = dat / (max_el - min_el) * (2);
			  this.param.curve.push(dat - 1);
          }
        } else {
          this.param.curve.push(value[i]);
        }
        this.filters[i].gain.setValueAtTime(
            this.param.range * 5 * this.param.curve[i], 0);
      }
    }
  },

  /**
   * The range of the EQ curve. <br>
   * Gain value for each filter is calculated: gain = curve[i] * 5 * range
   * @name Equalizer~range
   * @member
   * @type number
   */
  range: {
    get: function() { return this.param.range; },
    set: function(value) {
      this.param.range = value;
      this.curve = this.param.curve;
    }
  }
});
