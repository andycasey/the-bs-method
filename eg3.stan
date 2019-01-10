
data {
    int<lower=0> P; // number of pixels
    vector[P] flux;
    vector[P] wavelength;
    vector[P] flux_error;
    real line_wavelength_bounds[2];
    real small;
    real scalar;
    int<lower=0> continuum_order;
}

transformed data {
    real wavelength_range;
    real mean_flux_error;
    wavelength_range = (wavelength[P] - wavelength[1]);
    mean_flux_error = sum(flux_error)/P;
}

parameters {
    real continuum_coefficients[1 + continuum_order];

    real<lower=line_wavelength_bounds[1], upper=line_wavelength_bounds[2]> line_wavelength;
    
    real<lower=0, upper=1> line_amplitude;
    real<lower=0.01> line_sigma;
    real<lower=-10, upper=5> log_outlier_sigma;
    real<lower=log(scalar*mean_flux_error) + pow(exp(log_outlier_sigma), 2)> outlier_mean;
    real<lower=(10*line_sigma)/wavelength_range, upper=1> theta;

}


transformed parameters {

    real outlier_sigma;
    vector[P] continuum;
    vector[P] model_flux;
    
    outlier_sigma = exp(log_outlier_sigma);

    // do continuum
    continuum = rep_vector(continuum_coefficients[1], P);

    
    for (i in 1:continuum_order)
        continuum = continuum + wavelength * pow(continuum_coefficients[i + 1], i);
    

    // do model flux
    {
        vector[P] chi;
        vector[P] absorption;

        chi = (wavelength - rep_vector(line_wavelength, P))/line_sigma;
        absorption = 1 - line_amplitude * exp(-0.5 * chi .* chi);
  
        model_flux = continuum .* absorption;
    }

}

model {
    
    /*
    for (p in 1:P)
        target += normal_lpdf(flux[p] | model_flux[p], flux_error[p]);
    */
    
    for (p in 1:P) {
        real v[2];
        v[1] = model_flux[p] - flux[p];
        v[2] = small;

        target += log_mix(theta,
                          normal_lpdf(flux[p] | model_flux[p], flux_error[p]),
                          lognormal_lpdf(max(v) | outlier_mean, outlier_sigma));
    }
}


generated quantities {
  vector[P] p_outlier;

  for (p in 1:P) {
      real lps[2];
      real v[2];
      v[1] = model_flux[p] - flux[p];
      v[2] = small;

      lps[1] = log(theta) + normal_lpdf(flux[p] | model_flux[p], flux_error[p]);
      lps[2] = log1m(theta) + lognormal_lpdf(max(v) | outlier_mean, outlier_sigma);

      p_outlier[p] = exp(lps[2] - log_sum_exp(lps[1], lps[2]));
  }

}

