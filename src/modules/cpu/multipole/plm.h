
double Plm(const int l, const int m, const double x);

// The relationship has been found:
//
// P[n,l,x] (((-1)^Abs[l + 1] Pochhammer[-n, l] Pochhammer[n, 1 + l])/n)^-1 = P[n,-l,x]
//
// this function transforms a Plm into Pl(-m)
double Plm_negate_order(const int l, const int m, const double plm);
