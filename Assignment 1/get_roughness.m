function rough = get_roughness(w, phi_tilda, lambda)

rough = (lambda/2)*(w.'*phi_tilda*w);

end