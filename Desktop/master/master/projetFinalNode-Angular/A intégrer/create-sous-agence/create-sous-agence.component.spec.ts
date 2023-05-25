import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CreateSousAgenceComponent } from './create-sous-agence.component';

describe('CreateSousAgenceComponent', () => {
  let component: CreateSousAgenceComponent;
  let fixture: ComponentFixture<CreateSousAgenceComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CreateSousAgenceComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CreateSousAgenceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
