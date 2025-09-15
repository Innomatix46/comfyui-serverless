"""Pagination utilities."""
from typing import List, Dict, Any, Optional, TypeVar, Generic
from pydantic import BaseModel
from sqlalchemy.orm import Query
from math import ceil

T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response model."""
    items: List[T]
    total: int
    page: int
    per_page: int
    pages: int
    has_prev: bool
    has_next: bool
    prev_num: Optional[int] = None
    next_num: Optional[int] = None


def paginate(
    query: Query,
    page: int = 1,
    per_page: int = 20,
    max_per_page: int = 100
) -> Dict[str, Any]:
    """Paginate a SQLAlchemy query."""
    # Validate parameters
    page = max(1, page)
    per_page = min(max(1, per_page), max_per_page)
    
    # Get total count
    total = query.count()
    
    # Calculate pagination info
    pages = ceil(total / per_page)
    has_prev = page > 1
    has_next = page < pages
    prev_num = page - 1 if has_prev else None
    next_num = page + 1 if has_next else None
    
    # Get items for current page
    offset = (page - 1) * per_page
    items = query.offset(offset).limit(per_page).all()
    
    return {
        'items': items,
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': pages,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_num': prev_num,
        'next_num': next_num
    }


def paginate_list(
    items: List[T],
    page: int = 1,
    per_page: int = 20,
    max_per_page: int = 100
) -> Dict[str, Any]:
    """Paginate a list of items."""
    # Validate parameters
    page = max(1, page)
    per_page = min(max(1, per_page), max_per_page)
    
    total = len(items)
    
    # Calculate pagination info
    pages = ceil(total / per_page)
    has_prev = page > 1
    has_next = page < pages
    prev_num = page - 1 if has_prev else None
    next_num = page + 1 if has_next else None
    
    # Get items for current page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_items = items[start_idx:end_idx]
    
    return {
        'items': page_items,
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': pages,
        'has_prev': has_prev,
        'has_next': has_next,
        'prev_num': prev_num,
        'next_num': next_num
    }


def get_pagination_params(
    page: Optional[int] = None,
    per_page: Optional[int] = None,
    default_per_page: int = 20,
    max_per_page: int = 100
) -> tuple[int, int]:
    """Get validated pagination parameters."""
    page = max(1, page or 1)
    per_page = min(max(1, per_page or default_per_page), max_per_page)
    
    return page, per_page


class PaginationLinks:
    """Generate pagination links for API responses."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def generate_links(
        self,
        page: int,
        per_page: int,
        pages: int,
        has_prev: bool,
        has_next: bool,
        **query_params
    ) -> Dict[str, Optional[str]]:
        """Generate pagination links."""
        links = {
            'self': self._build_url(page, per_page, **query_params),
            'first': self._build_url(1, per_page, **query_params),
            'last': self._build_url(pages, per_page, **query_params),
            'prev': None,
            'next': None
        }
        
        if has_prev:
            links['prev'] = self._build_url(page - 1, per_page, **query_params)
        
        if has_next:
            links['next'] = self._build_url(page + 1, per_page, **query_params)
        
        return links
    
    def _build_url(self, page: int, per_page: int, **query_params) -> str:
        """Build URL with pagination parameters."""
        params = {
            'page': page,
            'per_page': per_page,
            **query_params
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Build query string
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        
        return f"{self.base_url}?{query_string}"