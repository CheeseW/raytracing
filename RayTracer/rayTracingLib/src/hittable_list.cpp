#include "hittable_list.h"

namespace rayUtilities {
	bool HittableList::hit(const Ray& r, const double tMin, const double tMax, HitRecord& rec) const
	{
		HitRecord temp;
		bool hit = false;
		auto closest = tMax;

		for (const auto& obj : objects) {
			if (obj->hit(r, tMin, closest, temp)) {
				hit = true;
				closest = temp.t;
				rec = temp;
			}
		}
		return hit;
	}
}