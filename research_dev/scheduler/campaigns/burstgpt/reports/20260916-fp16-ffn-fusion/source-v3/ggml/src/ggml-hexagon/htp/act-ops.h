#ifndef HTP_ACT_OPS_H
#define HTP_ACT_OPS_H

#include <stdint.h>

void htp_glu_f32(uint8_t * dst, const uint8_t * gate, const uint8_t * up, int nc, int op);

#endif
